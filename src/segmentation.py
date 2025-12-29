import torch
import logging
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


class FoodSegmenter:
    def __init__(self, model_path):
        logging.info("Preparing segmenter...")
        cfg = get_cfg()
        cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            )
        )

        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        cfg.MODEL.WEIGHTS = model_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

        self.predictor = DefaultPredictor(cfg)
        logging.info("Segmenter is ready.")

    def segment(self, image):
        outputs = self.predictor(image)
        instances = outputs["instances"].to("cpu")

        masks = instances.pred_masks.numpy()
        boxes = instances.pred_boxes.tensor.numpy()

        return masks, boxes
