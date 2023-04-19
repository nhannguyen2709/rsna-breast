import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import HfArgumentParser, TrainingArguments, set_seed
from transformers.trainer import logger
from transformers.trainer_utils import get_last_checkpoint, is_main_process

from data_args import DataArguments
from dataset import MultiImageCollator, MultiImageDataset
from engine_multi import CustomTrainer, compute_metrics
from metrics import pfbeta_metric
from model import MultiImageModel, build_model
from model_args import ModelArguments

torch.set_float32_matmul_precision("high")
torch.multiprocessing.set_sharing_strategy("file_system")


def main():
    parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
    data_args, model_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this"
                " behavior, change the `--output_dir` or add `--overwrite_output_dir` to train"
                " from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        # transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    # logger.info(f"Training/evaluation parameters {training_args}")
    # Set seed before initializing model.
    set_seed(training_args.seed)

    df = pd.read_csv("data/train_kaggle_boxes.csv")
    df = df[(df["patient_id"] != 822) & (df["image_id"] != 1942326353)]
    train_df = df[df["fold"] != data_args.fold]
    val_df = df[df["fold"] == data_args.fold]

    if data_args.use_external:
        train_df = pd.concat([train_df, pd.read_csv("data/train_csaw_boxes.csv")], 0).reset_index(
            drop=True
        )

    train_dataset = MultiImageDataset(train_df, "train", data_args)
    val_dataset = MultiImageDataset(val_df, "val", data_args)

    # Initialize trainer
    single_image_model = build_model(model_args.model_name)
    if training_args.do_train and last_checkpoint is None:
        assert model_args.resume_single != "", "Path to single image model checkpoint is empty"
        single_image_model.load_state_dict(torch.load(model_args.resume_single, "cpu"))

    model = MultiImageModel(single_image_model, transformer_dim=256, num_heads=4, dropout=0.5)

    if last_checkpoint is None and model_args.resume is not None:
        logger.info(f"Loading {model_args.resume} ...")
        checkpoint = torch.load(model_args.resume, "cpu")
        if "state_dict" in checkpoint:
            checkpoint = checkpoint.pop("state_dict")
        checkpoint = {
            k.replace("_orig_mod.", ""): v for k, v in checkpoint.items() if "aug." not in k
        }
        model.load_state_dict(checkpoint)

    if training_args.do_eval and model_args.swa and model_args.resume is None:
        checkpoint = {}
        assert model_args.resume_swa is not None
        checkpoints = model_args.resume_swa.split(" ")

        logger.info(f"Loading averaged checkpoints {checkpoints} ...")
        for ckpt_path in checkpoints:
            iter_ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
            for k, v in iter_ckpt.items():
                if "aug." in k:
                    continue
                if k not in checkpoint:
                    checkpoint[k] = v
                else:
                    checkpoint[k] += v
        checkpoint = {k: v / len(checkpoints) for k, v in checkpoint.items()}
        torch.save(checkpoint, os.path.join(training_args.output_dir, "swa_model.bin"))
        logger.info(f"Saved SWA checkpoint")
        model.load_state_dict(checkpoint)

    model = model.to(device="cuda")
    if training_args.torchdynamo == "inductor":
        torch.compile(model)
        training_args.torchdynamo = "eager"

    trainer = CustomTrainer(
        pos_neg_ratio=data_args.pn_ratio,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=MultiImageCollator(),
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        checkpoint = last_checkpoint if last_checkpoint else None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        output = trainer.predict(val_dataset, metric_key_prefix="eval")
        metrics = output.metrics
        torch.save(
            {
                "label_ids": output.label_ids[0],
                "patient_ids": output.label_ids[1],
                "laterality": output.label_ids[2],
                "predictions": output.predictions[0],
                "embeddings": output.predictions[1],
                "metrics": output.metrics,
            },
            os.path.join(training_args.output_dir, "output.pth"),
        )
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        logger.info("*** Optimize pF1 ***")
        thresholds = np.arange(0.1, 0.9, 0.01)
        scores = []
        predictions = torch.sigmoid(torch.from_numpy(output.predictions[0][:, 0])).numpy()
        labels = output.label_ids[0]
        patient_ids = output.label_ids[1]
        laterality = output.label_ids[2]
        pred_df = pd.concat(
            [
                pd.Series(patient_ids),
                pd.Series(laterality),
                pd.Series(labels),
                pd.Series(predictions),
            ],
            1,
        )
        pred_df.columns = ["patient_id", "laterality", "label", "pred"]
        pred_df["prediction_id"] = (
            pred_df["patient_id"].astype(str) + "_" + pred_df["laterality"].astype(str)
        )
        tmp = pred_df.groupby(["prediction_id"]).agg({"label": "max", "pred": "mean"})

        labels = tmp["label"].values
        predictions = tmp["pred"].values
        for threshold in thresholds:
            preds = predictions
            preds = preds > threshold
            pf1 = pfbeta_metric(labels, preds)
            scores.append(pf1)
        best_threshold = thresholds[np.argmax(scores)]
        best_score = np.max(scores)
        logger.info(f"Best pF1: {best_score} @ {best_threshold}")


if __name__ == "__main__":
    main()
