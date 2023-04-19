from PIL import Image
import torch
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
from transformers import DetrFeatureExtractor, DetrForObjectDetection

torch.set_grad_enabled(False)
# weights_path = "epoch19-step440/"
weights_path = "epoch11-step720/"

model = DetrForObjectDetection.from_pretrained(weights_path)
model = model.cuda()
model.eval()
feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")

# img_paths = glob("data/train_png_images/*/*")

dfv1 = pd.read_csv("data/cropv1.csv")
dfv1 = dfv1.sort_values(by=["conf"]).head(250)
img_paths = [
    os.path.join("data/train_png_images", str(pid), str(iid) + ".png")
    for pid, iid in zip(dfv1["patient_id"].values, dfv1["img_id"].values)
]


df = []

for img_path in tqdm(img_paths):
    image = Image.open(img_path).convert("RGB")
    encoding = feature_extractor(image, return_tensors="pt")
    encoding = {k: v.cuda(non_blocking=True) for k, v in encoding.items()}
    outputs = model(**encoding)
    target_sizes = torch.tensor(image.size[::-1]).unsqueeze(0)
    target_sizes = target_sizes.cuda(non_blocking=True)

    outputs = feature_extractor.post_process(outputs, target_sizes)

    scores = outputs[0]["scores"]
    box_score = scores.max().item()
    box = outputs[0]["boxes"][torch.argmax(scores)].tolist()

    w, h = image.size
    x0, y0, x1, y1 = box
    x0 = int(np.clip(x0, 0, w))
    x1 = int(np.clip(x1, 0, w))
    y0 = int(np.clip(y0, 0, h))
    y1 = int(np.clip(y1, 0, h))

    img_id = img_path.split("/")[-1].split(".")[0]
    patient_id = img_path.split("/")[-2]
    box_str = " ".join([str(x0), str(y0), str(x1), str(y1)])
    df.append([img_id, patient_id, box_str, box_score])

    img_fn = img_path.split("/")[-1]
    Image.fromarray(np.array(image)[y0:y1, x0:x1]).save(img_fn)

df = pd.DataFrame(df)
df.columns = ["img_id", "patient_id", "bbox", "conf"]
df.to_csv("data/crop.csv", index=False)
