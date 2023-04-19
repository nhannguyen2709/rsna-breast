from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from transformers import Trainer
from transformers.trainer_pt_utils import nested_detach

from metrics import pfbeta_metric
from samplers import ProportionalTwoClassesBatchSampler


class CustomTrainer(Trainer):
    def __init__(self, pos_neg_ratio=1, **kwargs):
        super().__init__(**kwargs)
        self.pos_neg_ratio = pos_neg_ratio

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        pos_bsize = self.args.train_batch_size // (self.pos_neg_ratio + 1)
        return ProportionalTwoClassesBatchSampler(
            self.train_dataset.df["cancer"].values,
            self.args.train_batch_size,
            minority_size_in_batch=pos_bsize,
        )

    def compute_loss(
        self, model: torch.nn.Module, inputs: Dict[str, torch.Tensor], return_outputs=False
    ):
        loss, logits, embeddings = model(
            inputs["images"].cuda(non_blocking=True), inputs["labels"].cuda(non_blocking=True)
        )

        if return_outputs:
            return (loss, logits, embeddings)
        return loss

    def create_optimizer(self):
        model = self.model
        no_decay = []
        for n, m in model.named_modules():
            if isinstance(
                m,
                (
                    torch.nn.BatchNorm1d,
                    torch.nn.BatchNorm2d,
                    torch.nn.LayerNorm,
                    torch.nn.GroupNorm,
                ),
            ):
                no_decay.append(n)

        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, logits, embeddings = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()

        if prediction_loss_only:
            return (loss, None, None)
        logits = logits.float()
        logits = nested_detach(logits)
        embeddings = embeddings.float()
        embeddings = nested_detach(embeddings)
        del inputs["images"]
        return (
            loss,
            (logits, embeddings),
            (inputs["labels"], inputs["patient_ids"], inputs["laterality"], inputs["view"]),
        )


def compute_metrics(eval_preds):
    predictions = torch.sigmoid(torch.from_numpy(eval_preds.predictions[0])).numpy().reshape(-1)
    isnan = np.isnan(predictions)
    labels = eval_preds.label_ids[0]
    patient_ids = eval_preds.label_ids[1]
    laterality = eval_preds.label_ids[2]
    # remove NaN
    labels = labels[~isnan]
    predictions = predictions[~isnan]
    patient_ids = patient_ids[~isnan]
    laterality = laterality[~isnan]

    idx2lat = {0: "L", 1: "R"}
    pred_df = pd.concat(
        [pd.Series(patient_ids), pd.Series(laterality), pd.Series(labels), pd.Series(predictions)],
        1,
    )
    pred_df.columns = ["patient_id", "laterality", "label", "pred"]
    pred_df["laterality"] = pred_df["laterality"].apply(lambda x: idx2lat.get(x))
    pred_df["prediction_id"] = pred_df["patient_id"].astype(str) + "_" + pred_df["laterality"]
    pred_df = pred_df.groupby(["prediction_id"]).agg({"label": "max", "pred": "mean"})

    image_pf1 = pfbeta_metric(
        labels,
        predictions,
    )
    image_auc = roc_auc_score(labels, predictions)
    patient_pf1 = pfbeta_metric(pred_df["label"].values, pred_df["pred"].values)
    patient_auc = roc_auc_score(pred_df["label"].values, pred_df["pred"].values)
    return {
        "image_pF1": image_pf1,
        "image_AUC": image_auc,
        "patient_pF1": patient_pf1,
        "patient_auc": patient_auc,
    }
