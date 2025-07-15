import tkinter as tk
from tkinter import filedialog, Label
from PIL import Image, ImageTk
import torch
from torchvision import transforms
import os

# === Загрузка модели ===
model = torch.load("C:/Users/lego/vs code projects/mushroom_classifier/model.pth", map_location=torch.device("cpu"), weights_only=False)
model.eval()

# === Классы ===
classes = sorted(os.listdir("C:/Users/lego/vs code projects/mushroom_classifier/dataset"))

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
        img = Image.open(file_path).resize((224, 224))
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk

        result = predict_image(file_path)
        result_label.config(text=f"Результат: {result}")

# === Окно ===
window = tk.Tk()
window.title("Классификация грибов")
window.geometry("300x400")

btn = tk.Button(window, text="Загрузить фото гриба", command=open_image)
btn.pack(pady=10)

image_label = Label(window)
image_label.pack()

result_label = Label(window, text="Результат появится здесь")
result_label.pack(pady=10)

window.mainloop()