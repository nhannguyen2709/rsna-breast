import os
from glob import glob
from typing import List

import numpy as np
from joblib import Parallel, delayed
from PIL import Image

from preprocessing import parser, resize_png


def resize_and_save_png(src_path: str, tgt_path: str, size: List[int]):
    img = np.array(Image.open(src_path))
    img = resize_png(img, size)
    Image.fromarray(img).save(tgt_path)


if __name__ == "__main__":
    args = parser.parse_args()
    src_folder = args.src
    tgt_folder = args.tgt
    size = [int(str_s) for str_s in args.size.split(" ")]
    n_jobs = os.cpu_count()
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
        delayed(resize_and_save_png)(src_path, tgt_path, size)
        for src_path, tgt_path in zip(src_paths, tgt_paths)
    )
