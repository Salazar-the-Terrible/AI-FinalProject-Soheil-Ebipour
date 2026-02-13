import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os

IMG_SIZE = 128
DEVICE = "cpu"

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,   16,  3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16,  32,  3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32,  64,  3, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64,  128, 3, padding=1)
        self.bn4   = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn5   = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(256 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))
        x = self.pool(torch.relu(self.bn5(self.conv5(x))))

        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

model = SimpleCNN()
model.load_state_dict(torch.load("cnn_cat_dog.pth", map_location=DEVICE))
model.eval()

CLASS_NAMES = ["cat", "dog"] 


transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


def preprocess_image(path):
    img = Image.open(path).convert("RGB")
    tensor = transform(img)
    tensor = tensor.unsqueeze(0)
    return tensor, img

def predict_image():
    global img_tk

    path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if not path:
        return

    tensor, original_img = preprocess_image(path)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    label = CLASS_NAMES[pred.item()]
    confidence = conf.item()

    # Show image
    original_img = original_img.resize((300, 300))
    img_tk = ImageTk.PhotoImage(original_img)
    image_label.config(image=img_tk)

    result_label.config(
        text=f"Prediction: {label.upper()}  ({confidence:.2f})"
    )


root = tk.Tk()
root.title("Cat vs Dog Classifier (PyTorch)")
root.geometry("420x520")

title = Label(root, text="Cat vs Dog CNN", font=("Arial", 18))
title.pack(pady=10)

image_label = Label(root)
image_label.pack(pady=10)

browse_button = Button(
    root,
    text="Browse Image",
    font=("Arial", 12),
    command=predict_image
)
browse_button.pack(pady=20)

result_label = Label(root, text="Prediction:", font=("Arial", 14))
result_label.pack(pady=10)

root.mainloop()
