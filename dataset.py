import json
import os
import warnings
from ast import literal_eval
from typing import Dict, List, Optional

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from data_args import DataArguments


def add_max_pad_breast_shape(df: pd.DataFrame):
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore")
        df["xmin"] = df["xmin"].clip(lower=0).astype(int)
        df["xmax"] = df["xmax"].astype(int)
        df["ymin"] = df["ymin"].clip(lower=0).astype(int)
        df["ymax"] = df["ymax"].astype(int)
        df["h"] = df["ymax"] - df["ymin"]
        df["w"] = df["xmax"] - df["xmin"]
    df = df.merge(
        df.groupby("patient_id")
        .apply(lambda x: [x["h"].max(), x["w"].max()])
        .to_frame("max_pad_breast_shape")
        .reset_index(),
        on=["patient_id"],
    ).reset_index(drop=True)
    return df


def apply_StratifiedGroupKFold(X, y, groups, n_splits, random_state=42):
    df_out = X.copy(deep=True)
    # split
    cv = StratifiedGroupKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    for fold_index, (train_index, val_index) in enumerate(cv.split(X, y, groups)):
        df_out.loc[val_index, "fold"] = fold_index
        # check
        train_groups, val_groups = groups[train_index], groups[val_index]
        assert len(set(train_groups) & set(val_groups)) == 0
    df_out = df_out.astype({"fold": "int64"})
    return df_out


def build_transforms(
    mode: str = "train", image_height: int = 1536, image_width: int = 1024, flip: bool = True
):
    if mode == "train":
        img_bbox_transform = A.ShiftScaleRotate(0.1, 0.2, 20, p=0.8)
        if flip:
            transform = [A.HorizontalFlip(), A.VerticalFlip()]
        else:
            transform = []
        transform.extend(
            [
                A.OneOf(
                    [
                        # A.RandomGamma(gamma_limit=(50, 150)),
                        A.RandomBrightnessContrast(),
                    ],
                    p=1,
                ),
                A.CoarseDropout(
                    5,
                    int(0.2 * image_height),
                    int(0.2 * image_width),
                    min_height=int(0.1 * image_height),
                    min_width=int(0.1 * image_width),
                ),
                A.Normalize(mean=0, std=1),
                ToTensorV2(),
            ]
        )
        transform = A.Compose(transform)
    else:
        img_bbox_transform = None
        transform = A.Compose([A.Normalize(mean=0, std=1), ToTensorV2()])
    return img_bbox_transform, transform


def read_img(path):
    return np.array(Image.open(path))


def read_and_crop_img_with_breast_bbox(
    info: pd.Series,
    img_bbox_transform: Optional[A.DualTransform] = None,
    image_height: int = 1536,
    image_width: int = 1024,
):
    # get bbox
    xmin, ymin, xmax, ymax = info["xmin"], info["ymin"], info["xmax"], info["ymax"]
    mh, mw = info["max_pad_breast_shape"]
    # augment original image
    img = read_img(info.png_file)
    h, w = img.shape
    if img_bbox_transform is not None:
        transformed = img_bbox_transform(image=img, bboxes=[[xmin, ymin, xmax, ymax, 0]])
        img = transformed["image"]
        bbox = list(transformed["bboxes"][0][:4])
        bbox[0] = int(max(bbox[0], 0))
        bbox[1] = int(max(bbox[1], 0))
        bbox[2] = int(min(bbox[2], w))
        bbox[3] = int(min(bbox[3], h))
        # print([xmin, ymin, xmax, ymax], bbox)
        xmin, ymin, xmax, ymax = bbox
        if xmin >= xmax or ymin >= ymax:
            xmin, ymin, xmax, ymax = info["xmin"], info["ymin"], info["xmax"], info["ymax"]
    # crop and pad zeros
    crop = img[ymin:ymax, xmin:xmax]
    image = np.zeros((image_height, image_width), np.uint8)
    scale = min(image_height / mh, image_width / mw)
    dsize = (
        min(image_width, int(scale * crop.shape[1])),
        min(image_height, int(scale * crop.shape[0])),
    )
    if dsize != (crop.shape[1], crop.shape[0]):
        crop = cv2.resize(crop, dsize=dsize, interpolation=cv2.INTER_LINEAR)
    ch, cw = crop.shape
    x = (image_width - cw) // 2
    y = (image_height - ch) // 2
    image[y : y + ch, x : x + cw] = crop
    # print(info.png_file, crop.shape, image.shape)
    return image


class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        mode: str,
        data_args: DataArguments,
    ):
        super().__init__()

        self.df = add_max_pad_breast_shape(df)
        self.image_height = data_args.height
        self.image_width = data_args.width
        self.img_bbox_transform, self.transform = build_transforms(
            mode, self.image_height, self.image_width
        )
        # print(self.img_bbox_transform, self.transform)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        info = self.df.iloc[idx]
        image = read_and_crop_img_with_breast_bbox(
            info, self.img_bbox_transform, self.image_height, self.image_width
        )
        image = self.transform(image=image)["image"]
        return {
            "images": image,
            "labels": info["cancer"],
            "patient_ids": info["patient_id"],
            "laterality": 0 if info["laterality"] == "L" else 1,
            "view": 0 if info["view"] == "CC" else 1,
            "file_name": info.png_file.split("/")[-1],
            "idxs": idx,
        }


class MultiImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        mode: str,
        data_args: DataArguments,
    ):
        super().__init__()

        self.df = add_max_pad_breast_shape(df)
        self.df["prediction_id"] = (
            self.df["patient_id"].astype(str)
            + "_"
            + self.df["laterality"]
            + "_"
            + self.df["png_file"].apply(lambda x: x.split("/")[1])
        )
        self.labels_df = self.df.groupby(["prediction_id"]).agg({"cancer": "max"}).reset_index()
        self.image_height = data_args.height
        self.image_width = data_args.width
        self.img_bbox_transform, self.transform = build_transforms(
            mode, self.image_height, self.image_width, flip=False
        )

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        info = self.labels_df.iloc[idx]
        df = self.df[self.df["prediction_id"] == info["prediction_id"]]
        images = []
        for _, info in df.iterrows():
            image = read_and_crop_img_with_breast_bbox(
                info, self.img_bbox_transform, self.image_height, self.image_width
            )
            image = self.transform(image=image)["image"]
            images.append(image)
        images = torch.stack(images)
        return {
            "images": images,
            "lengths": images.shape[0],
            "patient_ids": info["patient_id"],
            "laterality": 0 if info["laterality"] == "L" else 1,
            "labels": df["cancer"].max(),
        }


class SingleImageCollator:
    def __call__(self, features):
        images, labels, patient_ids, laterality, view, idxs = [], [], [], [], [], []

        for f in features:
            images.append(f["images"])
            labels.append(f["labels"])
            patient_ids.append(f["patient_ids"])
            laterality.append(f["laterality"])
            view.append(f["view"])
            idxs.append(f["idxs"])

        images = torch.stack(images)
        labels = torch.as_tensor(labels)
        patient_ids = torch.as_tensor(patient_ids)
        laterality = torch.as_tensor(laterality)
        view = torch.as_tensor(view)
        idxs = torch.as_tensor(idxs)

        batch = {
            "images": images,
            "labels": labels,
            "patient_ids": patient_ids,
            "laterality": laterality,
            "view": view,
            "idxs": idxs,
        }
        return batch


class MultiImageCollator:
    def __call__(self, features):
        images, lengths, labels, patient_ids, laterality = [], [], [], [], []
        for f in features:
            images.append(f["images"])
            lengths.append(f["lengths"])
            labels.append(f["labels"])
            patient_ids.append(f["patient_ids"])
            laterality.append(f["laterality"])

        images = torch.cat(images)
        labels = torch.as_tensor(labels)
        patient_ids = torch.as_tensor(patient_ids)
        laterality = torch.as_tensor(laterality)
        batch = {
            "images": images,
            "lengths": lengths,
            "labels": labels,
            "patient_ids": patient_ids,
            "laterality": laterality,
        }
        return batch
