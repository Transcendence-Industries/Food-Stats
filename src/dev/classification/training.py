import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataset import get_dataloaders
from torchvision.models import efficientnet_b0

device = "cuda" if torch.cuda.is_available() else "cpu"

train_loader, test_loader, num_classes = get_dataloaders()

model = efficientnet_b0(pretrained=True)
model.classifier[1] = nn.Linear(1280, 101)  # num_classes = 101
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(10):
    model.train()
    total_loss = 0

    for imgs, labels in tqdm(train_loader):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1} | Loss: {total_loss:.3f}")

output_dir = "./data/classification/model"
os.makedirs(output_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(output_dir, "model_final.pth"))

val_loss = 0
correct = 0
total = 0

with torch.no_grad():
    for imgs, labels in tqdm(test_loader):
        imgs, labels = imgs.to(device), labels.to(device)

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        val_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

val_acc = correct / total

print(f"Validation Loss: {val_loss} | Validation Accuracy: {val_acc}")
