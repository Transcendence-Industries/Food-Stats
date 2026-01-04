import os
import io
import json
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

INPUT_DIR = "./data/food101/data"
OUTPUT_DIR = "./data/classification"


def parquet_to_images_and_labels(split):
    output_dir_images = os.path.join(OUTPUT_DIR, split, "images")
    output_dir_labels = os.path.join(OUTPUT_DIR, split)
    os.makedirs(output_dir_images, exist_ok=True)
    os.makedirs(output_dir_labels, exist_ok=True)

    labels_json = {}

    for file in os.listdir(INPUT_DIR):
        if file.startswith(split) and file.endswith(".parquet"):
            df = pd.read_parquet(os.path.join(INPUT_DIR, file))

            for _, row in tqdm(df.iterrows(), total=len(df)):
                image_data = row["image"]["bytes"]
                image_name = row["image"]["path"]
                label = row["label"]
                image_path = os.path.join(output_dir_images, image_name)

                labels_json[image_name] = label

                if not os.path.exists(image_path):
                    if isinstance(image_data, (bytes, bytearray)):
                        img = Image.open(io.BytesIO(image_data))
                    elif isinstance(image_data, list):
                        arr = np.array(image_data, dtype=np.uint8)
                        img = Image.fromarray(arr)
                    else:
                        raise ValueError("Unknown image data type")

                    img.save(image_path)

    with open(os.path.join(output_dir_labels, "labels.json"), "w") as file:
        json.dump(labels_json, file, indent=4)


if __name__ == "__main__":
    parquet_to_images_and_labels("train")
    parquet_to_images_and_labels("validation")
