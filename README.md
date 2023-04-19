# 6th place solution for RSNA Screening Mammography Breast Cancer Detection Challenge

Repository to accompany with discussion forum post (https://www.kaggle.com/competitions/rsna-breast-cancer-detection/discussion/391979).

## Pre-processing

* Download the dataset from Kaggle and unzip to `data/` folder
```
kaggle competitions download -c rsna-breast-cancer-detection
mkdir data/
unzip rsna-breast-cancer-detection.zip -d data/
```

* Extract pixel arrays from DICOM files, resize with aspect-ratio preserving and save to disk
```
python dicom_pipeline.py data/train_images data/train_png_images --src-csv data/train.csv --size 1536
```

* Crop breast RoI using YOLOX detector
```
python crop_roi.py data/train_png_images --src-csv data/train.csv --tgt-csv data/train_kaggle_yolox.csv
```

## Pre-training


## Training and evaluation