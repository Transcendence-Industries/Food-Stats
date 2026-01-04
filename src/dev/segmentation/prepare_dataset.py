import os
import io
import json
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

INPUT_DIR = "./data/FoodSeg103/data"
OUTPUT_DIR = "./data/segmentation"


def parquet_to_images_and_masks(split):
    output_dir_images = os.path.join(OUTPUT_DIR, split, "images")
    output_dir_masks = os.path.join(OUTPUT_DIR, split, "masks")
    os.makedirs(output_dir_images, exist_ok=True)
    os.makedirs(output_dir_masks, exist_ok=True)

    for file in os.listdir(INPUT_DIR):
        if file.startswith(split) and file.endswith(".parquet"):
            df = pd.read_parquet(os.path.join(INPUT_DIR, file))

            for _, row in tqdm(df.iterrows(), total=len(df)):
                image_data = row["image"]["bytes"]
                mask_data = row["label"]["bytes"]
                image_name = row["image"]["path"]
                image_path = os.path.join(output_dir_images, image_name)
                mask_path = os.path.join(output_dir_masks, image_name)

                if not os.path.exists(image_path):
                    if isinstance(image_data, (bytes, bytearray)):
                        img = Image.open(io.BytesIO(image_data))
                    elif isinstance(image_data, list):
                        arr = np.array(image_data, dtype=np.uint8)
                        img = Image.fromarray(arr)
                    else:
                        raise ValueError("Unknown image data type")

                    img.save(image_path)

                if not os.path.exists(mask_path):
                    if isinstance(mask_data, (bytes, bytearray)):
                        img = Image.open(io.BytesIO(mask_data))
                    elif isinstance(mask_data, list):
                        arr = np.array(mask_data, dtype=np.uint8)
                        img = Image.fromarray(arr)
                    else:
                        raise ValueError("Unknown image data type")

                    img.save(mask_path)


def mask_to_coco(mask, image_id, anno_id_start):
    annotations = []
    anno_id = anno_id_start

    # Any non-zero pixel is food
    bin_mask = (mask > 0).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for c in contours:
        if cv2.contourArea(c) < 100:
            continue

        seg = c.flatten().tolist()
        x, y, w_box, h_box = cv2.boundingRect(c)
        area = cv2.contourArea(c)

        annotations.append({
            "id": anno_id,
            "image_id": image_id,
            "category_id": 0,  # SINGLE CLASS
            "segmentation": [seg],
            "area": float(area),
            "bbox": [float(x), float(y), float(w_box), float(h_box)],
            "iscrowd": 0,
        })

        anno_id += 1

    return annotations


def images_and_masks_to_annotations(split):
    output_file = os.path.join(OUTPUT_DIR, split, "annotations.json")
    img_dir = os.path.join(OUTPUT_DIR, split, "images")
    msk_dir = os.path.join(OUTPUT_DIR, split, "masks")
    imgs = os.listdir(img_dir)

    if os.path.exists(output_file):
        return

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": "food"}],
    }

    anno_id = 1
    for i, file_name in tqdm(enumerate(sorted(imgs)), total=len(imgs)):
        img_path = os.path.join(img_dir, file_name)
        mask_path = os.path.join(msk_dir, file_name)

        img = Image.open(img_path)
        w, h = img.size

        coco["images"].append({
            "id": i,
            "file_name": file_name,
            "height": h,
            "width": w
        })

        mask = np.array(Image.open(mask_path))
        annos = mask_to_coco(mask, image_id=i, anno_id_start=anno_id)
        coco["annotations"].extend(annos)
        anno_id += len(annos)

    with open(output_file, "w") as f:
        json.dump(coco, f, indent=4)


if __name__ == "__main__":
    parquet_to_images_and_masks("train")
    parquet_to_images_and_masks("validation")

    images_and_masks_to_annotations("train")
    images_and_masks_to_annotations("validation")
