import argparse
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from yolox.data.data_augment import preproc

from yolox_infer import Predictor

is_cuda_avail = torch.cuda.is_available()
if is_cuda_avail:
    from torch.cuda import amp
torch.multiprocessing.set_sharing_strategy("file_system")
parser = argparse.ArgumentParser(description="predict masks")
parser.add_argument("src", help="source folder")
parser.add_argument("--src-csv", help="source csv")
parser.add_argument("--tgt-csv", help="target csv")
args = parser.parse_args()


class CustomDataset(Dataset):
    def __init__(self, path_list, input_size=512):
        self.path_list = path_list
        self.input_size = input_size

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, i):
        image = cv2.imread(self.path_list[i])
        image, ratio = preproc(image, self.input_size)
        image = torch.from_numpy(image)
        return image, ratio, self.path_list[i]


def collate_fn(batch):
    imgs = torch.stack([item[0] for item in batch])
    ratios = [item[1] for item in batch]
    paths = [item[2] for item in batch]
    return imgs, ratios, paths


if __name__ == "__main__":
    # model = torch.hub.load(
    #     "../yolov5",
    #     "custom",
    #     path="../yolov5/runs/train/exp/weights/best.pt",
    #     source="local",
    #     force_reload=True,
    # )
    # model.amp = True
    # model.cuda()
    model = Predictor()

    dataset = CustomDataset(
        [path for path in glob(f"{args.src}/**", recursive=True) if path.endswith(".png")],
        input_size=model.test_size,
    )
    dataloader = DataLoader(dataset, batch_size=128, num_workers=16, collate_fn=collate_fn)

    src_df = pd.read_csv(args.src_csv)
    tgt_df = []
    img_paths = []

    for images, ratios, paths in tqdm(dataloader):
        outputs = model(images, ratios)
        for output, orig_img, path in zip(outputs, images, paths):
            if len(output) > 0:
                xyxy = output[0].tolist()[0]
                tgt_df.append(
                    pd.DataFrame(
                        {
                            "xmin": xyxy[0],
                            "ymin": xyxy[1],
                            "xmax": xyxy[2],
                            "ymax": xyxy[3],
                            "confidence": output[1][0].item(),
                            "class": [0],
                            "name": ["breast"],
                        }
                    )
                )
                img_paths.append(path)
            else:
                print("Cannot detect ", path)

        # outputs = model(images, size=320)
        # xyxy = outputs.pandas().xyxy
        # for orig_img, img_path, pred_df in zip(images, paths, xyxy):
        #     if len(pred_df) > 0:
        #         tgt_df.append(pred_df.head(1))
        #         img_paths.append(img_path)
        #     else:
        #         print("Cannot detect ", img_path)
        #         tgt_df.append(
        #             pd.DataFrame(
        #                 {
        #                     "xmin": [0],
        #                     "ymin": [0],
        #                     "xmax": [orig_img.shape[1]],
        #                     "ymax": [orig_img.shape[0]],
        #                     "confidence": [0],
        #                     "class": [0],
        #                     "name": ["breast"],
        #                 }
        #             )
        #         )

    tgt_df = pd.concat(tgt_df).reset_index(drop=True).drop(columns=["class", "name"])
    tgt_df["png_file"] = img_paths

    src_df = pd.read_csv(args.src_csv)
    # tgt_df = pd.read_csv(args.tgt_csv)
    # tgt_df = tgt_df[["xmin", "ymin", "xmax", "ymax", "confidence", "png_file"]]

    # from dataset import apply_StratifiedGroupKFold

    # src_df = apply_StratifiedGroupKFold(src_df, src_df["cancer"], src_df["patient_id"], 5)

    csv_filename = args.src_csv.split("/")[-1]
    if csv_filename == "train.csv":
        src_df["key"] = src_df["patient_id"].astype(str) + "_" + src_df["image_id"].astype(str)
        if "fold" not in src_df:
            from dataset import apply_StratifiedGroupKFold

            src_df = apply_StratifiedGroupKFold(src_df, src_df["cancer"], src_df["patient_id"], 5)
        tgt_df["key"] = tgt_df["png_file"].apply(lambda x: x.split("/")[-1].split(".")[0])
    
    cols = ["key", "patient_id", "image_id", "laterality", "view", "cancer"]
    if "fold" in src_df:
        cols.append("fold")

    tgt_df = tgt_df.merge(src_df[cols], on=["key"])
    tgt_df.drop(columns=["key"]).to_csv(args.tgt_csv, index=False)
