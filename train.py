import os
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np
from PIL import Image
import pandas as pd

# Параметры
data_dir = "dataset"
epochs = 50
batch_size = 128
lr = 0.0001
output_dir = "training_results"
os.makedirs(output_dir, exist_ok=True)

# ✅ Улучшенная аугментация
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
])

# Для валидации используем просто ресайз и нормализацию
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Разделяем датасет на train и val
full_dataset = datasets.ImageFolder(data_dir)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

# Применяем трансформы
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform

# Даталоадеры
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Классы
class_names = full_dataset.classes
print("Классы:", class_names)

# Модель
model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, len(class_names))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Обучение
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Для сохранения метрик
history = {'train_loss': [], 'val_loss': [], 'accuracy': []}


def save_predictions_images(images, labels, preds, epoch):
    """Сохраняет примеры предсказаний с подписями"""
    plt.figure(figsize=(12, 8))
    for i in range(min(6, len(images))):
        plt.subplot(2, 3, i + 1)
        image = images[i].cpu().numpy().transpose((1, 2, 0))
        image = np.clip(image, 0, 1)

        true_label = class_names[labels[i]]
        pred_label = class_names[preds[i]]

        plt.imshow(image)
        plt.title(f"True: {true_label}\nPred: {pred_label}", color='green' if true_label == pred_label else 'red')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/epoch_{epoch + 1}_predictions.png")
    plt.close()


def evaluate_model(epoch):
    """Оценка модели на валидационном наборе"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    val_images = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Сохраняем для анализа
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Сохраняем несколько изображений для визуализации
            if len(val_images) < 6:
                val_images.extend(inputs.cpu()[:6 - len(val_images)])

    # Сохраняем примеры предсказаний
    save_predictions_images(val_images, all_labels[:6], all_preds[:6], epoch)

    # Вычисляем метрики
    val_loss = val_loss / len(val_loader)
    accuracy = correct / total

    # Сохраняем confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - Epoch {epoch + 1}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f"{output_dir}/epoch_{epoch + 1}_confusion_matrix.png")
    plt.close()

    # Сохраняем classification report
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f"{output_dir}/epoch_{epoch + 1}_classification_report.csv")

    model.train()
    return val_loss, accuracy


for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    val_loss, accuracy = evaluate_model(epoch)

    # Сохраняем метрики
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['accuracy'].append(accuracy)

    print(
        f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")

    # Сохраняем графики метрик
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Accuracy', color='green')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/metrics_epoch_{epoch + 1}.png")
    plt.close()

# Сохраняем модель и историю обучения
torch.save(model.state_dict(), f"{output_dir}/model.pth")
pd.DataFrame(history).to_csv(f"{output_dir}/training_history.csv")

print(f"✅ Обучение завершено. Результаты сохранены в {output_dir}")