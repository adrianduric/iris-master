import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Setting random seed for testing
RANDOM_SEED = 77
torch.manual_seed(RANDOM_SEED)

# Setting hyperparameters
config = {
          'batch_size': 16,
          'use_cuda': True,
          'epochs': 10,
          'learningRate': 1e-3
         }

# Task 1a)
# Storing paths to images and corresponding labels
root_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = "mandatory1_data"

img_paths = []
img_labels = []

i = 0
for label_dir, _, filenames in os.walk(os.path.join(root_dir, images_dir)):
    label = i

    for filename in filenames:
        file_path = os.path.join(label_dir, filename)
        
        img_paths.append(file_path)
        img_labels.append(label)

# Performing split of dataset
# First splitting test set and the rest
TEST_PORTION = 3000/len(img_paths) # splits approx. 3000 of features to test set
img_paths_temp, img_paths_test, img_labels_temp, img_labels_test = train_test_split(img_paths, img_labels, test_size=TEST_PORTION, stratify=img_labels)

# Then splitting training and validation sets
VAL_PORTION = 2000/len(img_paths_temp) # approx. 2000 features in validation set
img_paths_train, img_paths_val, img_labels_train, img_labels_val = train_test_split(img_paths_temp, img_labels_temp, test_size=VAL_PORTION, stratify=img_labels_temp)

# Asserting that sets are disjoint
assert set(img_paths_train).isdisjoint(set(img_paths_val))
assert set(img_paths_train).isdisjoint(set(img_paths_test))
assert set(img_paths_val).isdisjoint(set(img_paths_test))


# Task 1b)
# Creating Dataset class for image files
class ImageDataset(Dataset):
    def __init__(self, img_paths, img_labels):
        self.img_paths = img_paths
        self.img_labels = img_labels
        self.transform = ResNet18_Weights.DEFAULT.transforms()

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx])
        label = self.img_labels[idx]
        image = self.transform(image)

        return image, label
#end ImageDataset

# Creating Datasets and DataLoaders
train_dataset = ImageDataset(img_paths_train, img_labels_train)
val_dataset = ImageDataset(img_paths_val, img_labels_val)
test_dataset = ImageDataset(img_paths_test, img_labels_test)

train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=True)


# Task 1c)
# Loading ResNet18 pretrained with weights
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

# Swapping last layer for classification of 6 labels
model.fc = nn.Linear(in_features=512, out_features=6, bias=True)

if config["use_cuda"]:
    model.to("cuda")

# Choosing CrossEntropyLoss as loss function
loss_fn = nn.CrossEntropyLoss()

# Choosing Adam as optimizer
optimizer = torch.optim.SGD(model.parameters())


# Task 1e)
# Train model for one epoch
def train_model(model, dataloader):

    # Iterating through batches
    for batch_idx, (batch_images, batch_labels) in enumerate(tqdm(dataloader)):
        if config["use_cuda"]:
            batch_images = batch_images.to("cuda")
            batch_labels = batch_labels.to("cuda")
            
        # Forward pass and loss calculation
        batch_predictions  = model(batch_images)
        loss = loss_fn(batch_predictions, batch_labels)
        
        # Backpropagation and updating weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
#end train_model

# Test model on testing or validation data
def test_model(model, dataloader):

    all_predictions = torch.Tensor()
    all_labels = torch.Tensor()
    if config["use_cuda"]:
        all_predictions = all_predictions.to("cuda")
        all_labels = all_labels.to("cuda")

    # Iterating through batches
    for batch_idx, (batch_images, batch_labels) in enumerate(tqdm(dataloader)):
        if config["use_cuda"]:
            batch_images = batch_images.to("cuda")
            batch_labels = batch_labels.to("cuda")
        with torch.no_grad():
            print(f"batch labels shape: {batch_predictions.shape}")
            batch_predictions = model(batch_images)
            print(f"batch preds shape: {batch_predictions.shape}")
            batch_images = nn.functional.one_hot(batch_images)
            print(f"batch preds shape after one hot: {batch_predictions.shape}")
            all_predictions = torch.cat((all_predictions, batch_predictions), 0)
            all_labels = torch.cat((all_labels, batch_labels), 0)

    # Task 1d)
    # Calculating performance metrics
    all_predictions = all_predictions.to("cpu")
    all_labels = all_labels.to("cpu")
    
    # accuracy = accuracy_score(all_labels, all_predictions)
    ap_score = average_precision_score(all_labels, all_predictions, average=None)
    mean_ap_score = average_precision_score(all_labels, all_predictions, average="macro")
        
    return  ap_score, mean_ap_score
#end test_model
# Train model for specified amount of epochs
for e in range(config["epochs"]):
    print(f"----------- EPOCH {e+1} -----------")
    train_model(model, train_dataloader)

    # Tracking metrics on validation sets during training
    ap_score, mean_ap_score = test_model(model, val_dataloader)
    print(f"Accuracy: {accuracy}, AP: {ap_score}, mAP: {mean_ap_score}")











