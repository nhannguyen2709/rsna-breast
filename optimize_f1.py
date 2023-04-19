import gc
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

from engine import pfbeta_metric

idx2lat = {0: "L", 1: "R"}
idx2view = {0: "CC", 1: "MLO"}
folds = list(range(int(sys.argv[1])))
output_dir = sys.argv[2]

image_labels = []
patient_ids = []
laterality = []
probs = []
embeddings = []
view = []

for i in folds:
    output: Dict[str, np.ndarray] = torch.load(
        os.path.join(f"{output_dir}-fold{i}", "output.pth"), "cpu"
    )
    image_labels.append(output["label_ids"])
    patient_ids.append(output["patient_ids"])
    laterality.append(output["laterality"])
    probs.append(torch.sigmoid(torch.from_numpy(output["predictions"])).view(-1).numpy())
    embeddings.append(output["embeddings"])
    if "view" in output:
        view.append(output["view"])

if len(view) > 0:
    view = np.concatenate(view, 0)
image_labels = np.concatenate(image_labels, 0)
embeddings = np.concatenate(embeddings, 0)
laterality = np.concatenate(laterality, 0)
probs = np.concatenate(probs, 0)
patient_ids = np.concatenate(patient_ids, 0)

pred_df = pd.DataFrame(
    {"patient_id": patient_ids, "laterality": laterality, "label": image_labels, "pred": probs}
)
pred_df["laterality"] = pred_df["laterality"].apply(lambda x: idx2lat.get(x))
pred_df["prediction_id"] = pred_df["patient_id"].astype(str) + "_" + pred_df["laterality"]

tmp = pred_df.groupby(["prediction_id"]).agg({"label": "max", "pred": "mean"})
labels = tmp["label"].values
predictions = tmp["pred"].values
tmp.reset_index().to_csv(output_dir + ".csv", index=False)
print(labels.shape, predictions.shape)


thresholds = np.arange(0.1, 0.9, 0.01)
scores = []
for threshold in thresholds:
    preds = predictions > threshold
    pf1 = pfbeta_metric(labels, preds)
    scores.append(pf1)

best_threshold = thresholds[np.argmax(scores)]
best_score = np.max(scores)

non_optimized_pf1 = pfbeta_metric(labels, predictions)
print(f"Load outputs from {output_dir} folds {folds}")

best_pf1 = pfbeta_metric(labels, predictions > best_threshold)
best_prec = precision_score(labels, predictions > best_threshold)
best_recall = recall_score(labels, predictions > best_threshold)
best_auc = roc_auc_score(labels, predictions)
print(
    f"thresh: {best_threshold} - pf1: {non_optimized_pf1} - binned pf1: {best_pf1} - precision: {best_prec} - recall: {best_recall} - auc: {best_auc}"
)
print(classification_report(labels, predictions > best_threshold))
print(confusion_matrix(labels, predictions > best_threshold))
