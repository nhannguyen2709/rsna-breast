import os
import torch
import torch.nn as nn
from transformers import DetrFeatureExtractor, DetrForObjectDetection
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pydicom
from typing import List
import gc
from tqdm import tqdm
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
from joblib import Parallel, delayed
import timm
import pandas as pd

torch.set_grad_enabled(False)


def convert_dicom_to_png(dicom_file: str) -> np.ndarray:
    """
    dicom_file: path to the dicom fife

    return
        gray scale image with pixel intensity in the range [0,255]
        None if cannot convert

    """
    data = pydicom.read_file(dicom_file)
    img = data.pixel_array

    img = (img - img.min()) / (img.max() - img.min())
    if data.PhotometricInterpretation == "MONOCHROME1":
        img = 1 - img
    img = (img * 255).astype(np.uint8)

    return img


class TestDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        info = self.df.iloc[idx]
        pred_id = info["prediction_id"]
        img = convert_dicom_to_png(info["path"])
        img = np.stack([img, img, img], -1)
        return img, pred_id


def collate_fn(batch):
    images = [item[0] for item in batch]
    pred_ids = [item[1] for item in batch]
    return images, pred_ids


def extract_bbox(output, h, w):
    x0, y0, x1, y1 = output["boxes"][torch.argmax(output["scores"])].tolist()
    x0 = int(np.clip(x0, 0, w))
    x1 = int(np.clip(x1, 0, w))
    y0 = int(np.clip(y0, 0, h))
    y1 = int(np.clip(y1, 0, h))
    return x0, y0, x1, y1


class ImageModel(nn.Module):
    def __init__(self, model_name: str = "base"):
        super().__init__()

        self.backbone = timm.create_model(model_name, pretrained=False, in_chans=1)
        self.backbone.reset_classifier(0, "avg")
        self.linear = nn.Linear(self.backbone.num_features, 1)

    def forward(self, images, labels=None):
        features = self.backbone(images)
        out = self.linear(features)
        return out


transform = Compose(
    [
        Resize(1024, 1024),
        Normalize(0, 1),
        ToTensorV2(),
    ]
)


def crop_and_resize(image: np.ndarray, bbox: List[int]):
    x0, y0, x1, y1 = bbox
    image = image[y0:y1, x0:x1, 0]
    image = transform(image=image)["image"]
    return image


IMAGES_DIR = "data/test_images"
WEIGHTS_DIR = "./"
test_df = pd.read_csv("data/test.csv")
patient_ids = test_df["patient_id"].unique()

detector = DetrForObjectDetection.from_pretrained(
    os.path.join(WEIGHTS_DIR, "epoch11-step720/"), use_pretrained_backbone=False
)
detector = detector.cuda()
detector.eval()
feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")

cls_model = ImageModel("tf_efficientnetv2_s_in21ft1k")
ckpt = torch.load(
    os.path.join(
        WEIGHTS_DIR, "efficientnet_v2s", "checkpoint-15037", "pytorch_model.bin"
    ),
    "cpu",
)
ckpt = {k.replace("_orig_mod.", ""): v for k, v in ckpt.items()}
cls_model.load_state_dict(ckpt)
cls_model = cls_model.cuda()
cls_model.eval()

sub_df = []
for patient_id in tqdm(patient_ids):
    df = test_df[test_df["patient_id"] == patient_id]
    df["path"] = df.apply(
        lambda x: os.path.join(
            IMAGES_DIR, str(x["patient_id"]), str(x["image_id"]) + ".dcm"
        ),
        axis=1,
    )
    test_dataset = TestDataset(df)
    test_loader = DataLoader(
        test_dataset, batch_size=16, num_workers=4, collate_fn=collate_fn
    )

    patient_probs = []

    for images, pred_ids in test_loader:
        encoding = feature_extractor(images, return_tensors="pt")
        encoding = {k: v.cuda(non_blocking=True) for k, v in encoding.items()}
        outputs = detector(**encoding)
        target_sizes = torch.stack([torch.tensor(image.shape[:-1]) for image in images])
        target_sizes = target_sizes.cuda(non_blocking=True)
        outputs = feature_extractor.post_process(outputs, target_sizes)
        # extract boxes
        bboxes = [
            extract_bbox(output, target_size[0].item(), target_size[1].item())
            for output, target_size in zip(outputs, target_sizes)
        ]
        del encoding, outputs
        gc.collect()
        torch.cuda.empty_cache()
        # crop and resize
        images = Parallel(n_jobs=4)(
            delayed(crop_and_resize)(image, bbox) for image, bbox in zip(images, bboxes)
        )
        images = torch.stack(images).cuda(non_blocking=True)
        # classify
        logits = cls_model(images)
        probs = torch.sigmoid(logits).view(-1)
        patient_probs.extend(probs.cpu().numpy().tolist())

    df["cancer"] = patient_probs
    sub_df.append(df.groupby(["prediction_id"]).agg({"cancer": "mean"}).reset_index())

sub_df = pd.concat(sub_df, axis=0).to_csv("submission.csv", index=False)
