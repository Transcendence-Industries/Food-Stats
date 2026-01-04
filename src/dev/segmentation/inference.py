import cv2
import numpy as np
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

DRAW_SEGMENTATION = True

cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
)

cfg.MODEL.DEVICE = "cpu"
cfg.MODEL.WEIGHTS = "./data/segmentation/model/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

predictor = DefaultPredictor(cfg)

image = cv2.imread("./data/samples/sample_3.jpg")
outputs = predictor(image)

scores = outputs["instances"].scores.cpu().numpy()
masks = outputs["instances"].pred_masks.cpu().numpy()
boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()

food_crops = []
for box in boxes:
    x1, y1, x2, y2 = box.astype(int)
    food_crops.append(image[y1:y2, x1:x2])

print(f"Detected {len(food_crops)} foods")

if DRAW_SEGMENTATION:
    output_img = image.copy()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int)

        cv2.rectangle(
            output_img,
            (x1, y1),
            (x2, y2),
            color=(0, 255, 0),
            thickness=2
        )

        label = f"Food: {scores[i]:.2f}"
        cv2.putText(
            output_img,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        mask = masks[i]
        colored_mask = np.zeros_like(image, dtype=np.uint8)
        colored_mask[mask] = (0, 255, 0)

        output_img = cv2.addWeighted(output_img, 1.0, colored_mask, 0.4, 0)

    cv2.imwrite("segmentation_result.jpg", output_img)
