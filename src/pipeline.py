import cv2
import logging
import numpy as np
from typing import List
from pydantic import BaseModel

from segmentation import FoodSegmenter
from classification import FoodClassifier
from database import CaloriesDatabase

SEGMENTER_MODEL_PATH = "./bin/segmentation/model_final.pth"
CLASSIFIER_MODEL_PATH = "./bin/classification/model_final.pth"
CLASS_MAPPING_PATH = "./bin/classification/mapping.json"
DATABASE_PATH = "./bin/database/db.json"


class FoodReport(BaseModel):
    label: str
    grams: float
    calories: float


class TotalReport(BaseModel):
    foods: List[FoodReport]
    total_grams: float
    total_calories: float


def estimate_grams_and_calories(mask, image_shape, cal_per_100g, serving_size):
    # mask: binary mask (H, W)
    # image_shape: (H, W, C)

    food_pixels = np.sum(mask)
    image_pixels = image_shape[0] * image_shape[1]

    area_ratio = food_pixels / image_pixels
    estimated_grams = area_ratio / 0.5 * serving_size
    estimated_calories = estimated_grams * cal_per_100g / 100

    return estimated_grams, estimated_calories


class MainPipeline:
    def __init__(self):
        logging.info("Preparing pipeline...")
        self.segmenter = FoodSegmenter(model_path=SEGMENTER_MODEL_PATH)
        self.classifier = FoodClassifier(model_path=CLASSIFIER_MODEL_PATH, class_mapping_path=CLASS_MAPPING_PATH)
        self.database = CaloriesDatabase(db_path=DATABASE_PATH)
        logging.info("Pipeline is ready.")

    def run(self, image_bytes):
        logging.debug("Running pipeline...")
        foods = []
        total_grams = 0
        total_calories = 0

        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        masks, boxes = self.segmenter.segment(image)
        logging.debug(f"Found {len(masks)} foods in the image.")

        for mask, box in zip(masks, boxes):
            x1, y1, x2, y2 = box.astype(int)
            crop = image[y1:y2, x1:x2]

            label = self.classifier.classify(crop)
            cal_per_100g = self.database.get_calories_per_100g(label)
            serving_size = self.database.get_serving_size(label)

            grams, calories = estimate_grams_and_calories(mask, image.shape, cal_per_100g, serving_size)

            total_grams += grams
            total_calories += calories
            foods.append(FoodReport(
                label=label,
                grams=round(grams, 1),
                calories=round(calories, 1),
            ))
            logging.debug(f"> {label}: {grams}g, {calories}kcal")

        logging.debug("Successfully ran pipeline.")
        return TotalReport(
            foods=foods,
            total_grams=round(total_grams, 1),
            total_calories=round(total_calories, 1),
        )
