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
root_dir = "~/iris-master/in4310/innleveringer/oblig1"
images_dir = "mandatory1_data"

img_paths = []
img_labels = []

i = 0
for label_dir, _, filenames in os.walk(os.path.join(root_dir, images_dir)):
    label = i
    print(label_dir)

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
        # self.transform = transforms.Compose([
        #     transforms.Resize(256),
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx])
        label = self.img_labels[idx]
        image = self.transform(image)
        return image, label

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

# Choosing CrossEntropyLoss as loss function
loss_fn = nn.CrossEntropyLoss()

# Choosing Adam as optimizer
optimizer = torch.optim.Adam(model.parameters())


# Task 1e)
# Code to run model in training and inference
def run_model(model, dataloader, is_training=False):
    all_predictions = torch.Tensor()
    all_labels = torch.Tensor()

    for batch_idx, (batch_images, batch_labels) in enumerate(tqdm(dataloader)):
        if not is_training:
            with torch.no_grad():
                batch_predictions = model(batch_images)
                all_predictions = torch.cat((all_predictions, batch_predictions), 0)
                all_labels = torch.cat((all_labels, batch_labels), 0)

        elif is_training:
            batch_predictions  = model(batch_images)
            loss = loss_fn(batch_predictions, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Task 1d)
    # Calculating performance metrics
    if not is_training:
        accuracy = accuracy_score(all_labels, all_predictions)
        ap_score = average_precision_score(all_labels, all_predictions, average=None)
        mean_ap_score = average_precision_score(all_labels, all_predictions, average="macro")
        
        return accuracy, ap_score, mean_ap_score

# Run model for specified amount of epochs
for e in range(config["epochs"]):
    run_model(model, train_dataloader, is_training=True)












