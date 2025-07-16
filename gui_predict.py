import tkinter as tk
from tkinter import filedialog, Label, Frame
from PIL import Image, ImageTk
import torch
from torchvision import transforms
import os
import json

# === Загрузка модели ===
model = torch.load("model.pth", map_location=torch.device("cpu"), weights_only=False)
model.eval()

# === Загрузка данных о грибах ===
with open('mushrooms.json', 'r', encoding='utf-8') as f:
    mushrooms_data = json.load(f)

# Создаем словарь для быстрого доступа к данным по названию гриба
mushrooms_info = {mushroom['name']: mushroom for mushroom in mushrooms_data['mushrooms']}

# === Классы ===
classes = sorted(os.listdir("dataset"))

# === Предобработка изображения ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def predict_image(img_path):
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        return classes[predicted.item()]


# === Интерфейс ===
def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg")])
    if file_path:
        # Отображение изображения
        img = Image.open(file_path).resize((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk

        # Классификация гриба
        result = predict_image(file_path)
        result_label.config(text=f"Распознанный гриб: {result}", font=('Arial', 14, 'bold'))

        # Отображение информации о грибе
        display_mushroom_info(result)


def display_mushroom_info(mushroom_name):
    # Очищаем предыдущую информацию
    for widget in info_frame.winfo_children():
        widget.destroy()

    if mushroom_name in mushrooms_info:
        info = mushrooms_info[mushroom_name]

        # Создаем элементы для отображения информации
        tk.Label(info_frame, text="Информация о грибе", font=('Arial', 12, 'bold')).pack(pady=(10, 5))

        # Создаем фрейм для информации
        data_frame = Frame(info_frame)
        data_frame.pack(padx=20, pady=5, fill=tk.X)

        # Отображаем информацию
        tk.Label(data_frame, text="Сезон:", font=('Arial', 10, 'bold')).pack(anchor='w')
        tk.Label(data_frame, text=info['season'], font=('Arial', 10)).pack(anchor='w', padx=20)

        tk.Label(data_frame, text="Место обитания:", font=('Arial', 10, 'bold')).pack(anchor='w', pady=(5, 0))
        tk.Label(data_frame, text=info['habitat'], font=('Arial', 10), wraplength=700, justify='left').pack(anchor='w',
                                                                                                            padx=20)

        tk.Label(data_frame, text="Симптомы отравления:", font=('Arial', 10, 'bold')).pack(anchor='w', pady=(5, 0))
        tk.Label(data_frame, text=info['symptoms'], font=('Arial', 10), wraplength=700, justify='left').pack(anchor='w',
                                                                                                             padx=20)

        tk.Label(data_frame, text="Информация о токсинах:", font=('Arial', 10, 'bold')).pack(anchor='w', pady=(5, 0))
        tk.Label(data_frame, text=info['toxin info'], font=('Arial', 10), wraplength=700, justify='left').pack(
            anchor='w', padx=20)

    else:
        tk.Label(info_frame, text=f"Нет данных о грибе '{mushroom_name}'", font=('Arial', 10)).pack()

    # Обновляем размер окна
    window.update_idletasks()
    window.geometry(f"800x{max(600, window.winfo_reqheight())}")


# === Создание основного окна ===
window = tk.Tk()
window.title("Классификация грибов")
window.geometry("800x600")
window.minsize(800, 600)

# Основной фрейм с возможностью прокрутки
main_frame = Frame(window)
main_frame.pack(fill=tk.BOTH, expand=1)

# Верхняя часть с кнопкой и изображением
top_frame = Frame(main_frame)
top_frame.pack(fill=tk.X, pady=10)

btn = tk.Button(top_frame, text="Загрузить фото гриба", command=open_image, font=('Arial', 10))
btn.pack(pady=10)

image_label = Label(top_frame)
image_label.pack()

result_label = Label(top_frame, text="Результат появится здесь", font=('Arial', 12))
result_label.pack(pady=10)

# Фрейм для информации о грибе
info_frame = Frame(main_frame)
info_frame.pack(fill=tk.BOTH, expand=1)

window.mainloop()