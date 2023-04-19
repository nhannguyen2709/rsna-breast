import matplotlib.pyplot as plt
import pydicom
import numpy as np
from PIL import Image
from glob import glob
import os
from joblib import Parallel, delayed


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


def convert_v1(dicom_file: str) -> np.ndarray:
    data = pydicom.read_file(dicom_file)

    if (
        ("WindowCenter" not in data)
        or ("WindowWidth" not in data)
        or ("PhotometricInterpretation" not in data)
        or ("RescaleSlope" not in data)
        or ("RescaleIntercept" not in data)
    ):

        print(f"{dicom_file} DICOM file does not have required fields")
        return

    # intentType = data.data_element('PresentationIntentType').value
    # if ( str(intentType).split(' ')[-1]=='PROCESSING' ):
    #     print(f"{dicom_file} got processing file")
    #     return

    c = data.data_element("WindowCenter").value  # data[0x0028, 0x1050].value
    w = data.data_element("WindowWidth").value  # data[0x0028, 0x1051].value
    if type(c) == pydicom.multival.MultiValue:
        c = c[0]
        w = w[0]

    photometricInterpretation = data.data_element("PhotometricInterpretation").value

    try:
        a = data.pixel_array
    except:
        print(f"{dicom_file} Cannot get get pixel_array!")
        return

    slope = data.data_element("RescaleSlope").value
    intercept = data.data_element("RescaleIntercept").value
    a = a * slope + intercept

    try:
        pad_val = data.get("PixelPaddingValue")
        pad_limit = data.get("PixelPaddingRangeLimit", -99999)
        if pad_limit == -99999:
            mask_pad = a == pad_val
        else:
            if str(photometricInterpretation) == "MONOCHROME2":
                mask_pad = (a >= pad_val) & (a <= pad_limit)
            else:
                mask_pad = (a >= pad_limit) & (a <= pad_val)
    except:
        # Manually create padding mask
        # this is based on the assumption that padding values take majority of the histogram
        print(f"{dicom_file} has no PixelPaddingValue")
        a = a.astype(np.int)
        pixels, pixel_counts = np.unique(a, return_counts=True)
        sorted_idxs = np.argsort(pixel_counts)[::-1]
        sorted_pixel_counts = pixel_counts[sorted_idxs]
        sorted_pixels = pixels[sorted_idxs]
        mask_pad = a == sorted_pixels[0]
        try:
            # if the second most frequent value (if any) is significantly more frequent than the third then
            # it is also considered padding value
            if sorted_pixel_counts[1] > sorted_pixel_counts[2] * 10:
                mask_pad = np.logical_or(mask_pad, a == sorted_pixels[1])
                print(
                    f"{dicom_file} most frequent pixel values: {sorted_pixels[0]}; {sorted_pixels[1]}"
                )
        except:
            print(f"{dicom_file} most frequent pixel value {sorted_pixels[0]}")

    # apply window
    mm = c - 0.5 - (w - 1) / 2
    MM = c - 0.5 + (w - 1) / 2
    a[a < mm] = 0
    a[a > MM] = 255
    mask = (a >= mm) & (a <= MM)
    a[mask] = ((a[mask] - (c - 0.5)) / (w - 1) + 0.5) * 255

    if str(photometricInterpretation) == "MONOCHROME1":
        a = 255 - a

    a[mask_pad] = 0
    return a.astype(np.uint8)


def process_dicom(dicom_path: str):
    png_img = convert_dicom_to_png(dicom_path)
    if png_img is None:
        print("Failed to convert ", dicom_path)
        return

    img_path = dicom_path.replace("_images", "_png_images").replace(".dcm", ".png")
    img_folder = "/".join(img_path.split("/")[:-1])
    os.makedirs(img_folder, exist_ok=True)

    Image.fromarray(png_img).save(img_path)


if __name__ == "__main__":
    # dicom_path = "data/train_images/10006/1459541791.dcm"
    # png_img = convert_dicom_to_png(dicom_path)

    # plt.imsave("original.png", png_img)

    Parallel(n_jobs=os.cpu_count(), verbose=1)(
        delayed(process_dicom)(dicom_path)
        for dicom_path in glob("/shared/disk1/nhan/train_images/*/*")
    )
