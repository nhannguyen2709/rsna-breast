import argparse
import os
from glob import glob
from typing import List, Optional

import cv2
import dicomsdl
import numpy as np
import torch
from joblib import Parallel, delayed
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize

parser = argparse.ArgumentParser(description="pre-process dicoms")
parser.add_argument("src", help="source folder")
parser.add_argument("tgt", help="target folder")
parser.add_argument(
    "--apply-cropping", default=False, action="store_true", help="crop black regions"
)
parser.add_argument("--to-uint16", default=False, action="store_true", help="stored as 16-bit PNG")
parser.add_argument("--resize", default=False, action="store_true", help="resize")
parser.add_argument("--size", default="1024 1024", help="resized shape")


def convert_dicom_to_png(dicom_file: str, to_uint16: bool = False) -> np.ndarray:
    """
    dicom_file: path to the dicom fife

    return
        gray scale image with pixel intensity in the range [0,255]
        None if cannot convert

    """
    data = dicomsdl.open(dicom_file)
    img = data.pixelData()

    img = (img - img.min()) / (img.max() - img.min())
    if data.PhotometricInterpretation == "MONOCHROME1":
        img = 1 - img

    if to_uint16:
        img = (img * 65535).astype(np.uint16)
    else:
        img = (img * 255).astype(np.uint8)

    return img


def convert_and_window(dicom_file: str) -> np.ndarray:
    data = dicomsdl.open(dicom_file)
    a = data.pixelData()

    c = data.WindowCenter
    if isinstance(c, list):
        c = c[0]
    w = data.WindowWidth
    if isinstance(w, list):
        w = w[0]
    photometric_interpretation = data.PhotometricInterpretation
    slope = data.RescaleSlope
    intercept = data.RescaleIntercept
    if slope is None:
        slope = 1.0
    if intercept is None:
        intercept = 0.0
    a = a * slope + intercept

    try:
        pad_val = data.PixelPaddingValue
    except AttributeError:
        pad_val = None

    if pad_val is not None:
        # pad_limit = data.get("PixelPaddingRangeLimit", -99999)
        pad_limit = -99999
        if pad_limit == -99999:
            mask_pad = a == pad_val
        else:
            if str(photometric_interpretation) == "MONOCHROME2":
                mask_pad = (a >= pad_val) & (a <= pad_limit)
            else:
                mask_pad = (a >= pad_limit) & (a <= pad_val)
    else:
        # Manually create padding mask
        # this is based on the assumption that padding values take majority of the histogram
        # print(f"{dicom_file} has no PixelPaddingValue")
        a = a.astype(int)
        pixels, pixel_counts = np.unique(a, return_counts=True)
        sorted_idxs = np.argsort(pixel_counts)[::-1]
        sorted_pixel_counts = pixel_counts[sorted_idxs]
        sorted_pixels = pixels[sorted_idxs]
        mask_pad = a == sorted_pixels[0]
        # if the second most frequent value (if any) is significantly more frequent
        # than the third then it is also considered padding value
        if sorted_pixel_counts[1] > sorted_pixel_counts[2] * 10:
            mask_pad = np.logical_or(mask_pad, a == sorted_pixels[1])
        #     print(
        #         f"{dicom_file} most frequent pixel values: {sorted_pixels[0]}; {sorted_pixels[1]}"
        #     )
        # else:
        #     print(f"{dicom_file} most frequent pixel value {sorted_pixels[0]}")

    # apply window
    mm = c - 0.5 - (w - 1) / 2
    MM = c - 0.5 + (w - 1) / 2
    a[a < mm] = 0
    a[a > MM] = 255
    mask = (a >= mm) & (a <= MM)
    a[mask] = ((a[mask] - (c - 0.5)) / (w - 1) + 0.5) * 255

    if str(photometric_interpretation) == "MONOCHROME1":
        a = 255 - a

    a[mask_pad] = 0
    return a.astype(np.uint8)


def get_boundaries(image: np.ndarray) -> List[int]:
    """
    return y_min, y_max, x_min, x_max
    NOTE: x_max, y_max are exclusive,
    crop the image by img[y_min:y_max, x_min:x_max]
    """
    mask = image != 0
    bounds = []

    for ax in [1, 0]:
        region = mask.any(axis=ax)
        bounds.append((region.argmax(), mask.shape[1 - ax] - region[::-1].argmax()))

    return [int(bounds[0][0]), int(bounds[0][1]), int(bounds[1][0]), int(bounds[1][1])]


def resize_png(
    img: np.ndarray,
    size: List[int],
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
) -> np.ndarray:
    if img.dtype == np.uint8:
        img: Image.Image = resize(Image.fromarray(img), size, interpolation)
        img = np.array(img)
    else:
        img = img.astype(np.float32)
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        img = resize(img, size, interpolation, antialias=True)[0, 0]
        img = img.numpy().astype(np.uint16)
    return img


def process_dicom(
    src_path: str,
    tgt_path: str,
    apply_cropping: bool = False,
    apply_resize: bool = False,
    to_uint16: bool = False,
    debug: bool = False,
    size: Optional[List[int]] = None,
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
):
    img = convert_dicom_to_png(src_path, to_uint16)
    if debug:
        print(f"Image shape {img.shape} - Image dtype: {img.dtype}")

    if apply_cropping:
        ymin, ymax, xmin, xmax = get_boundaries(img)
        h, w = img.shape
        aspect_ratio = size[0] / size[1]

        xmin = int(xmin * w)
        xmax = int(xmax * w)
        ymin = int(ymin * h)
        ymax = int(ymax * h)

        resulting_crop_h = ymax - ymin
        resulting_crop_w = xmax - xmin

        if int(aspect_ratio * resulting_crop_w) < resulting_crop_h:  # increase width
            # how much more width do I need to add?
            needed_width = (resulting_crop_h // 2) - resulting_crop_w

            # can I meet width without going past image?
            if needed_width + xmax > w:
                # padding/resizing is required (I use padding, in this case)
                difference = xmax + needed_width - w

                img = cv2.copyMakeBorder(img, 0, 0, difference, 0, cv2.BORDER_CONSTANT, value=0)

                # no need to offset bbox since origin is top left

            # expand bbox by needed width
            xmax += needed_width

        elif int(aspect_ratio * resulting_crop_w) > resulting_crop_h:  # increase height
            # how much more height do I need to add?
            needed_height = (resulting_crop_w * 2) - resulting_crop_h

            # can I meet height without going past image?
            if ymin - needed_height < 0:
                # padding/resizing is required (I use padding in this case)
                difference = needed_height - ymin

                img = cv2.copyMakeBorder(img, difference, 0, 0, 0, cv2.BORDER_CONSTANT, value=0)

                # offset bbox values to new origin
                ymin += difference
                ymax += difference

            ymin -= needed_height

        img = img[ymin:ymax, xmin:xmax]

    if apply_resize:
        assert size is not None, "requires specified size to resize image."
        img = resize_png(img, size, interpolation)
        if debug:
            print(
                f"Image shape {img.shape} - Min pixel value {img.min()} "
                f"- Max pixel value {img.max()}\n"
            )

    img = Image.fromarray(img)
    img.save(tgt_path)
    del img
    return


if __name__ == "__main__":
    args = parser.parse_args()
    src_folder = args.src
    tgt_folder = args.tgt
    apply_cropping = args.apply_cropping
    apply_resize = args.resize
    to_uint16 = args.to_uint16
    size = [int(str_s) for str_s in args.size.split(" ")]
    n_jobs = os.cpu_count() // 64
    verbose = 1

    src_paths = glob(f"{src_folder}/*/*")
    tgt_paths = [
        src_path.replace(src_folder.split("/")[-1], tgt_folder.split("/")[-1]).replace(
            ".dcm", ".png"
        )
        for src_path in src_paths
    ]
    tgt_folders = np.unique(
        ["/".join(tgt_path.split("/")[:-1]) for tgt_path in tgt_paths]
    ).tolist()
    for folder in tgt_folders:
        os.makedirs(folder, exist_ok=True)

    Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(process_dicom)(
            src_path, tgt_path, apply_cropping, apply_resize, to_uint16, size=size
        )
        for src_path, tgt_path in zip(src_paths, tgt_paths)
    )
