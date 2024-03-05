import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.model_selection import train_test_split

# Setting random seed for testing
RANDOM_SEED = 77
torch.manual_seed(RANDOM_SEED)

# Setting hyperparameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
EPOCHS = 5

# Task 1a)
# Setting X and y as features and targets
images_dir = "mandatory1_data"

X = []
y = []

i = 0
for label_dir, _, filenames in os.walk(images_dir):
    label = i

    for filename in filenames:
        file_path = os.path.join(label_dir, filename)
        
        X.append(file_path)
        y.append(label)

# Performing split of dataset
# First splitting test set and the rest
TEST_PORTION = 3000/len(X) # splits approx. 3000 of features to test set
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=TEST_PORTION, random_state=RANDOM_SEED, stratify=y)

# Then splitting training and validation sets
VAL_PORTION = 2000/len(X_temp) # approx. 2000 features in validation set
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=VAL_PORTION, random_state=RANDOM_SEED, stratify=y_temp)

# Asserting that sets are disjointed
assert set(X_train).isdisjoint(set(X_val))
assert set(X_train).isdisjoint(set(X_test))
assert set(X_val).isdisjoint(set(X_test))

# Task 1b)
# Creating Dataset class for image files
class ImageDataset(Dataset):
    def __init__(self, img_paths, img_labels):
        self.img_paths = img_paths
        self.img_labels = img_labels
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]) # Transforms for ResNet: https://pytorch.org/hub/pytorch_vision_resnet/

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx])
        label = self.img_labels[idx]
        image = self.transform(image)
        return image, label

# Creating Datasets and DataLoaders
train_dataset = ImageDataset(X_train, y_train)
val_dataset = ImageDataset(X_val, y_val)
test_dataset = ImageDataset(X_test, y_test)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Task 1c)
# Loading ResNet50 pretrained with weights
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)

# Choosing CrossEntropyLoss as loss function
loss_fn = nn.CrossEntropyLoss()

# Choosing Adam as optimizer
optimizer = torch.optim.Adam(model.parameters())


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * BATCH_SIZE + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

for t in range(EPOCHS):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")










