import argparse
import os

import numpy as np
import pandas as pd
import pydicom
from joblib import Parallel, delayed
from PIL import Image
from torchvision.transforms.functional import resize

parser = argparse.ArgumentParser(description="pre-process dicoms")
parser.add_argument("src", help="source folder")
parser.add_argument("tgt", help="target folder")
parser.add_argument("--num-slices", default=3)
parser.add_argument("--resize", default=False, action="store_true", help="resize")
parser.add_argument("--size", default="1024 1024", help="resized shape")
args = parser.parse_args()


def process_dicom_dbt(src_path, tgt_folder, key, num_slices=1, apply_resize=False, size=None):
    data = pydicom.dcmread(src_path)
    img = data.pixel_array
    img = (img - img.min()) / (img.max() - img.min())
    if data.data_element("PhotometricInterpretation") == "MONOCHROME1":
        img = 1 - img
    img = (img * 255).astype(np.uint8)

    dcm_bbox = boxes[boxes["key"] == key]
    if len(dcm_bbox) > 0:
        center_idx = dcm_bbox["Slice"].values[0]
    else:
        center_idx = len(img) // 2

    tgt_paths = []
    if num_slices == 1:
        img = img[center_idx]
        tgt_path = os.path.join(tgt_folder, f"slice-{center_idx}.png")

        if apply_resize:
            img = resize(Image.fromarray(img), size)
        else:
            img = Image.fromarray(img)

        img.save(tgt_path)
        tgt_paths.append(tgt_path)
    else:
        window = num_slices // 2
        for slice_idx in range(center_idx - window, center_idx + window + 1):
            tgt_path = os.path.join(tgt_folder, f"slice-{slice_idx}.png")

            if apply_resize:
                slice_img = resize(Image.fromarray(img[slice_idx]), size)
            else:
                slice_img = Image.fromarray(slice_img)

            slice_img.save(tgt_path)
            tgt_paths.append(tgt_path)

    return tgt_paths, [key] * len(tgt_paths)


if __name__ == "__main__":
    apply_resize = args.resize
    size = [int(str_s) for str_s in args.size.split(" ")]
    n_jobs = os.cpu_count()

    labels = pd.read_csv("BCS-DBT labels-train-v2.csv")
    labels["key"] = labels["PatientID"] + "-" + labels["StudyUID"] + "-" + labels["View"]
    boxes = pd.read_csv("BCS-DBT boxes-train-v2.csv")
    boxes["key"] = boxes["PatientID"] + "-" + boxes["StudyUID"] + "-" + boxes["View"]
    file_paths = pd.read_csv("BCS-DBT file-paths-train-v2.csv")
    file_paths["key"] = (
        file_paths["PatientID"] + "-" + file_paths["StudyUID"] + "-" + file_paths["View"]
    )

    df = labels.merge(file_paths.drop(columns=["PatientID", "StudyUID", "View"]), on=["key"])
    # df = df[df["PatientID"] == "DBT-P00003"]

    src_paths = df["descriptive_path"].values
    tgt_folders = []
    for p in src_paths:
        t = "/".join(p.replace(args.src, args.tgt).split("/")[:-1])
        os.makedirs(t, exist_ok=True)
        tgt_folders.append(t)

    meta = Parallel(n_jobs, verbose=1)(
        delayed(process_dicom_dbt)(src_path, tgt_folder, key, args.num_slices, apply_resize, size)
        for src_path, tgt_folder, key in zip(src_paths, tgt_folders, df["key"].values)
    )
    img_paths = []
    keys = []
    for out in meta:
        img_paths.extend(out[0])
        keys.extend(out[1])

    train_df = pd.DataFrame({"key": keys, "image": img_paths})
    train_df = train_df.merge(df[["key", "PatientID", "View", "Cancer"]], on=["key"])

    train_df.columns = ["key", "image", "patient_id", "view", "cancer"]
    train_df["laterality"] = train_df["view"].apply(lambda x: x[0].upper())
    train_df["view"] = train_df["view"].apply(lambda x: x[1:].upper())

    csv_path = "data/train_bcs_dbt.csv"
    print(f"Saving to {csv_path}")
    train_df.drop(columns=["key"]).to_csv(csv_path, index=False)
