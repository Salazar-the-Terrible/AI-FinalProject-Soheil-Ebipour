import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix,
    roc_curve, roc_auc_score, auc          # ← added for ROC
)
import seaborn as sns


IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.001
DATA_DIR = "data"


train_transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    #transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
    transforms.RandAugment(),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.25),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

test_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

class SafeImageFolder(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
        except Exception:
            return self.__getitem__((index + 1) % len(self.samples))

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

dataset = SafeImageFolder(
    root=DATA_DIR,
    transform=None
)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_set, test_set = torch.utils.data.random_split(
    dataset, [train_size, test_size]
)

train_set.dataset.transform = train_transform
test_set.dataset.transform  = test_transform

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)


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

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=0.0001)


loss_history      = []
accuracy_history  = []
f1_history        = []
precision_history = []
recall_history    = []


for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    loss_history.append(avg_loss)


    model.eval()
    all_preds = []
    all_labels = []

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    acc       = correct / total
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall    = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1        = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    accuracy_history.append(acc)
    precision_history.append(precision)
    recall_history.append(recall)
    f1_history.append(f1)

    print(f"Epoch {epoch+1:2d}: "
          f"Loss={avg_loss:7.4f} | "
          f"Acc={acc:6.3f} | "
          f"Prec={precision:6.3f} | "
          f"Rec={recall:6.3f} | "
          f"F1={f1:6.3f}")


epochs = range(1, EPOCHS + 1)

plt.figure(figsize=(12, 9))

plt.subplot(2, 2, 1)
plt.plot(epochs, loss_history, 'b-o')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(epochs, accuracy_history, 'g-o')
plt.title("Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(epochs, precision_history, 'c-o', label="Precision")
plt.plot(epochs, recall_history,    'm-o', label="Recall")
plt.title("Precision & Recall (macro)")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(epochs, f1_history, 'r-o')
plt.title("F1-score (macro)")
plt.xlabel("Epoch")
plt.ylabel("F1")
plt.grid(True)

plt.tight_layout()
plt.show()


model.eval()
all_probs = []
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images) 
        probs = torch.softmax(outputs, dim=1)[:, 1] 
        _, preds = torch.max(outputs, 1)

        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


y_true  = np.array(all_labels)
y_score = np.array(all_probs)
y_pred  = np.array(all_preds)


cm = confusion_matrix(y_true, y_pred)

class_names = ['Negative', 'Positive']  # ← customize: ['Background', 'Cat'], ['No', 'Yes'], etc.

print("\nConfusion Matrix:")
print("                 Predicted")
print(f"                 {class_names[0]:<10} {class_names[1]:<10}")
print(f"True {class_names[0]:<6}   {cm[0,0]:<10} {cm[0,1]:<10}")
print(f"     {class_names[1]:<6}   {cm[1,0]:<10} {cm[1,1]:<10}")

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=class_names, yticklabels=class_names,
    cbar=True
)
plt.title("Confusion Matrix (Final Model)")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.show()


fpr, tpr, _ = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr) 

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

torch.save(model.state_dict(), "cnn_cat_dog.pth")