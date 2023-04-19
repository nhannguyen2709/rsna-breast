import argparse
import os
import shutil
import time
from glob import glob

import cv2
import dicomsdl
import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import pandas as pd
import pydicom
import torch
from joblib import Parallel, delayed
from nvidia.dali import pipeline_def
from nvidia.dali.plugin.pytorch import feed_ndarray, to_torch_type
from nvidia.dali.types import DALIDataType
from pandarallel import pandarallel
from PIL import Image
from pydicom.filebase import DicomBytesIO
from torchvision.transforms.functional import resize
from tqdm import tqdm

parser = argparse.ArgumentParser(description="pre-process dicoms")
parser.add_argument("src", help="source folder")
parser.add_argument("tgt", help="target folder")
parser.add_argument("--src-csv", help="source csv")
parser.add_argument("--size", default=1536, help="resized shape")


def get_transfer_syntax(file):
    dcmfile = pydicom.dcmread(file)
    return dcmfile.file_meta.TransferSyntaxUID


def convert_dicom_to_j2k(file, patient_id, image_id, save_folder=""):
    with open(file, "rb") as fp:
        raw = DicomBytesIO(fp.read())
        ds = pydicom.dcmread(raw)
    offset = ds.PixelData.find(
        b"\x00\x00\x00\x0C"
    )  # <---- the jpeg2000 header info we're looking for
    hackedbitstream = bytearray()
    hackedbitstream.extend(ds.PixelData[offset:])
    with open(save_folder + f"{patient_id}_{image_id}.jp2", "wb") as binary_file:
        binary_file.write(hackedbitstream)


@pipeline_def
def j2k_decode_pipeline(j2kfiles):
    jpegs, _ = fn.readers.file(files=j2kfiles)
    images = fn.experimental.decoders.image(
        jpegs, device="mixed", output_type=types.ANY_DATA, dtype=DALIDataType.UINT16
    )
    return images


def non_j2k_decode(dicom_file: str, patient_id: str, image_id: str):
    data = dicomsdl.open(dicom_file)
    img = data.pixelData()

    min_, max_ = img.min(), img.max()
    img = (img - min_) / (max_ - min_)
    if data.getPixelDataInfo()["PhotometricInterpretation"] == "MONOCHROME1":
        img = 1 - img

    img = (img * 255).astype(np.uint8)

    img: Image.Image = resize(
        Image.fromarray(img), (SIZE, int(img.shape[1] / img.shape[0] * SIZE)), antialias=True
    )
    img.save(os.path.join(SAVE_FOLDER, f"{patient_id}_{image_id}.png"))


if __name__ == "__main__":
    args = parser.parse_args()
    IMAGES_FOLDER = args.src
    SAVE_FOLDER = args.tgt
    SIZE = int(args.size)
    J2K_FOLDER = "/tmp/j2k/"

    os.makedirs(SAVE_FOLDER, exist_ok=True)
    n_jobs = os.cpu_count()
    pandarallel.initialize(progress_bar=True, nb_workers=16)

    train_df = pd.read_csv(args.src_csv)
    # train_df = train_df.sample(n=1000)
    csv_filename = args.src_csv.split("/")[-1]
    if csv_filename == "train.csv":
        train_df["image"] = (
            IMAGES_FOLDER
            + "/"
            + train_df["patient_id"].astype(str)
            + "/"
            + train_df["image_id"].astype(str)
            + ".dcm"
        )

    train_df["TransferSyntaxUID"] = train_df["image"].parallel_apply(
        lambda x: get_transfer_syntax(x)
    )

    j2k_df = train_df[train_df["TransferSyntaxUID"] == "1.2.840.10008.1.2.4.90"]
    j2k_images = j2k_df["image"].values
    j2k_patient_ids = j2k_df["patient_id"].values
    j2k_image_ids = j2k_df["image_id"].values
    non_j2k_df = train_df[train_df["TransferSyntaxUID"] != "1.2.840.10008.1.2.4.90"]
    non_j2k_images = non_j2k_df["image"].values
    non_j2k_patient_ids = non_j2k_df["patient_id"].values
    non_j2k_image_ids = non_j2k_df["image_id"].values

    start = time.time()
    N_CHUNKS = 128
    CHUNKS = [
        (len(j2k_images) / N_CHUNKS * k, len(j2k_images) / N_CHUNKS * (k + 1))
        for k in range(N_CHUNKS)
    ]
    CHUNKS = np.array(CHUNKS).astype(int)
    for chunk in tqdm(CHUNKS):
        os.makedirs(J2K_FOLDER, exist_ok=True)
        _ = Parallel(n_jobs=n_jobs)(
            delayed(convert_dicom_to_j2k)(img, patient_id, image_id, save_folder=J2K_FOLDER)
            for img, patient_id, image_id in zip(
                j2k_images[chunk[0] : chunk[1]],
                j2k_patient_ids[chunk[0] : chunk[1]],
                j2k_image_ids[chunk[0] : chunk[1]],
            )
        )

        j2kfiles = [
            os.path.join(J2K_FOLDER, f"{patient_id}_{image_id}.jp2")
            for patient_id, image_id in zip(
                j2k_patient_ids[chunk[0] : chunk[1]], j2k_image_ids[chunk[0] : chunk[1]]
            )
        ]
        if len(j2kfiles) == 0:
            continue
        pipe = j2k_decode_pipeline(j2kfiles, batch_size=1, num_threads=2, device_id=0, debug=True)
        pipe.build()
        for f, dicom_file in zip(j2kfiles, j2k_images[chunk[0] : chunk[1]]):
            patient, image = f.split("/")[-1][:-4].split("_")
            dicom = dicomsdl.open(dicom_file)
            # Dali -> Torch
            out = pipe.run()
            img = out[0][0]
            img_torch = torch.empty(img.shape(), dtype=torch.int16, device="cuda")
            feed_ndarray(img, img_torch, cuda_stream=torch.cuda.current_stream(device=0))
            img = img_torch.float()
            # Scale, resize, invert on GPU !
            min_, max_ = img.min(), img.max()
            img = (img - min_) / (max_ - min_)
            if dicom.getPixelDataInfo()["PhotometricInterpretation"] == "MONOCHROME1":
                img = 1 - img
            img = img * 255

            img = resize(
                img.view(1, 1, img.shape[0], img.shape[1]),
                (SIZE, int(img.shape[1] / img.shape[0] * SIZE)),
                antialias=True,
            )[0, 0]
            # Back to CPU + SAVE
            img = img.cpu().numpy().astype(np.uint8)
            cv2.imwrite(os.path.join(SAVE_FOLDER, f"{patient}_{image}.png"), img)
        shutil.rmtree(J2K_FOLDER)
    end = time.time()
    print(f"Num. JPEG2000 files: {len(j2k_images)} - processing time: {end - start} seconds")

    start = time.time()
    _ = Parallel(n_jobs=n_jobs)(
        delayed(non_j2k_decode)(img, patient_id, image_id)
        for img, patient_id, image_id in zip(
            tqdm(non_j2k_images), non_j2k_patient_ids, non_j2k_image_ids
        )
    )
    end = time.time()
    print(f"Num. non-JPEG2000 files: {len(non_j2k_df)} - processing time: {end - start} seconds")
