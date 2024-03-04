import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.model_selection import train_test_split

# Setting random seed for testing
RANDOM_SEED = 77
torch.manual_seed(RANDOM_SEED)

# Task 1a)
# Setting X and y as features and targets
images_dir = "mandatory1_data"

X = []
y = []

for label_dir, _, filenames in os.walk(images_dir):
    label = os.path.basename(label_dir)

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
        image = read_image(self.img_paths[idx])
        label = img_labels[idx]
        image = self.transform(image)
        return image, label

# Creating Datasets and DataLoaders
train_dataset = ImageDataset(X_train, y_train)
val_dataset = ImageDataset(X_val, y_val)
test_dataset = ImageDataset(X_test, y_test)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

# Task 1c)
# Loading ResNet50 pretrained with weights
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)

# Choosing CrossEntropyLoss as loss function
loss_fn = nn.CrossEntropyLoss()













