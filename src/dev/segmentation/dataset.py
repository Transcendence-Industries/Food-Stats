from detectron2.data.datasets import register_coco_instances


def register_food():
    register_coco_instances(
        "foodseg103_train",
        {},
        "data/segmentation/train/annotations.json",
        "data/segmentation/train/images"
    )

    register_coco_instances(
        "foodseg103_val",
        {},
        "data/segmentation/validation/annotations.json",
        "data/segmentation/validation/images"
    )
