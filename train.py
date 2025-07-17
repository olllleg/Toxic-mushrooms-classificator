import os
from torchvision import datasets, transforms, models
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

# Параметры
data_dir = "dataset"
epochs = 20
batch_size = 8
lr = 0.0001

# ✅ Улучшенная аугментация
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
])

# Датасет и загрузка
dataset = datasets.ImageFolder(data_dir, transform=train_transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Классы
class_names = dataset.classes
print("Классы:", class_names)

# Модель
model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, len(class_names))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Обучение
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader):.4f}")

# Сохраняем модель
torch.save(model, "model.pth")
print("✅ Модель сохранена как model.pth")
