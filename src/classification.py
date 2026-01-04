import json
import torch
import logging
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0


class FoodClassifier:
    def __init__(self, model_path, class_mapping_path):
        logging.info("Preparing classifier...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        with open(class_mapping_path, "r") as file:
            self.class_mapping = json.load(file)

        self.class_mapping = {v: k for k, v in self.class_mapping.items()}

        self.model = efficientnet_b0(pretrained=False)
        self.model.classifier[1] = torch.nn.Linear(1280, len(self.class_mapping))
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        logging.info("Classifier is ready.")

    def classify(self, crop):
        img = Image.fromarray(crop).convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            idx = logits.argmax(dim=1).item()

        return self.class_mapping[idx]
