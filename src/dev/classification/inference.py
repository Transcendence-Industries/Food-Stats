import torch
from PIL import Image
from torchvision.models import efficientnet_b0
from dataset import get_transform, get_labels_mapping

device = "cuda" if torch.cuda.is_available() else "cpu"

model = efficientnet_b0(pretrained=False)
model.classifier[1] = torch.nn.Linear(1280, 101)
model.load_state_dict(torch.load("./data/classification/model/model_final.pth"))
model.to(device)
model.eval()

transform = get_transform()
labels_mapping = get_labels_mapping()

img = Image.open("./data/samples/sample_1.jpg").convert("RGB")
x = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    preds = model(x)
    idx = preds.argmax(dim=1).item()

print("Predicted food:", labels_mapping[idx])
