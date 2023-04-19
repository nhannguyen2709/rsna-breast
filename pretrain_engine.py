from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from transformers import Trainer
from transformers.trainer_pt_utils import nested_detach

from model import PretrainModel


class CustomTrainer(Trainer):
    def compute_loss(
        self, model: PretrainModel, inputs: Dict[str, torch.Tensor], return_outputs=False
    ):
        outputs = model(inputs["images"].cuda(non_blocking=True))
        loss = (
            F.cross_entropy(outputs[0], inputs["birads"])
            + F.cross_entropy(outputs[1], inputs["density"])
            + F.binary_cross_entropy_with_logits(outputs[2], inputs["findings"].float())
        )
        if return_outputs:
            return loss, outputs
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
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()

        if prediction_loss_only:
            return (loss, None, None)
        outputs = nested_detach(outputs)
        del inputs["images"]
        return loss, outputs, (inputs["birads"], inputs["density"], inputs["findings"])


def compute_metrics(eval_preds):
    birads_probs = F.softmax(torch.from_numpy(eval_preds.predictions[0]).float(), dim=1)
    density_probs = F.softmax(torch.from_numpy(eval_preds.predictions[1]).float(), dim=1)
    # finding_probs = torch.sigmoid(torch.from_numpy(eval_preds.predictions[2]).float())
    birads_auc = roc_auc_score(
        label_binarize(eval_preds.label_ids[0], classes=np.arange(birads_probs.shape[1])),
        birads_probs,
        multi_class="ovo",
    )
    density_auc = roc_auc_score(
        label_binarize(eval_preds.label_ids[1], classes=np.arange(density_probs.shape[1])),
        density_probs,
        multi_class="ovo",
    )
    overall_auc = (birads_auc + density_auc) / 2
    return {
        "AUC": overall_auc,
        "birads_AUC": birads_auc,
        "density_AUC": density_auc,
    }
