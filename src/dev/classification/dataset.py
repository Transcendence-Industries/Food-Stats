import os
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class LocalDataset(Dataset):
    def __init__(self, images_dir, labels_json, transform=None):
        self.images_dir = images_dir
        self.transform = transform

        with open(labels_json, "r") as f:
            self.labels_dict = json.load(f)

        self.image_files = list(self.labels_dict.keys())
        self.num_classes = len(set(self.labels_dict.values()))

        # Limit the number of images
        # import random
        # self.image_files = random.sample(self.image_files, 1000)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        label = self.labels_dict[img_name]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_labels_mapping():
    mapping_file = "./data/classification/mapping.json"
    with open(mapping_file, "r") as f:
        labels_mapping = json.load(f)
    labels_mapping = {v: k for k, v in labels_mapping.items()}
    return labels_mapping


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


def get_dataloaders(batch_size=32, num_workers=4):
    transform = get_transform()

    train_dataset = LocalDataset(
        images_dir="./data/classification/train/images",
        labels_json="./data/classification/train/labels.json",
        transform=transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_dataset = LocalDataset(
        images_dir="./data/classification/validation/images",
        labels_json="./data/classification/validation/labels.json",
        transform=transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader, train_dataset.num_classes
