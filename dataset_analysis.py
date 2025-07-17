import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

root_dir = 'C:/Users/lego/PycharmProjects/Practice/mushroom_classifier/dataset'

class_counts = {}
widths = []
heights = []
sizes_per_class = {}

# Проход по классам и файлам
for class_name in os.listdir(root_dir):
    class_path = os.path.join(root_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    filenames = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    class_counts[class_name] = len(filenames)

    class_widths = []
    class_heights = []

    for filename in filenames:
        img_path = os.path.join(class_path, filename)

        with Image.open(img_path) as img:
            w, h = img.size
            widths.append(w)
            heights.append(h)
            class_widths.append(w)
            class_heights.append(h)

    sizes_per_class[class_name] = (class_widths, class_heights)

# Подсчёт статистики по размерам
min_width, max_width = min(widths), max(widths)
min_height, max_height = min(heights), max(heights)
mean_width, mean_height = np.mean(widths), np.mean(heights)

print("Количество изображений в каждом классе:")
for cls, count in class_counts.items():
    print(f"  {cls}: {count}")

print("\nРазмеры изображений:")
print(f"  Ширина: min={min_width}, max={max_width}, среднее={mean_width:.2f}")
print(f"  Высота: min={min_height}, max={max_height}, среднее={mean_height:.2f}")

# Визуализация

plt.figure(figsize=(14, 6))

# Гистограмма количества изображений по классам
plt.subplot(1, 2, 1)
classes = list(class_counts.keys())
counts = list(class_counts.values())
plt.bar(classes, counts)
plt.xticks(rotation=45, ha='right')
plt.title('Количество изображений по классам')
plt.ylabel('Количество')
plt.legend()

# Гистограмма размеров (ширина и высота)
plt.subplot(1, 2, 2)
plt.hist(widths, bins=30, alpha=0.7, label='Ширина')
# синий
plt.hist(heights, bins=30, alpha=0.7, label='Высота')
plt.title('Распределение размеров изображений')
plt.xlabel('Размер (пиксели)')
plt.ylabel('Количество')

plt.tight_layout()
plt.show()