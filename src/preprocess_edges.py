import cv2
import os
import numpy as np

IMG_SIZE = 64
DATA_DIR = "data"

X = []
y = []

def load_images(label_name, label_value):
    folder = os.path.join(DATA_DIR, label_name)
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        edges = cv2.Canny(img, 100, 200)
        edges = edges / 255.0

        X.append(edges.flatten())
        y.append(label_value)

load_images("cat", 0)
load_images("dog", 1)

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)
