import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models import ResNet18_Weights
from sklearn.model_selection import train_test_split


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

        return image, label, idx
#end ImageDataset
    

# Task 1a)
# Storing paths to images and corresponding labels
def prepare_data(config, seed=None):
    root_path = os.path.dirname(os.path.abspath(__file__))
    images_dir = "mandatory1_data"
    images_path = os.path.join(root_path, images_dir)

    img_paths = []
    img_labels = []

    i = 0
    for label_dir in sorted(os.listdir(images_path)):
        label = i
        label_path = os.path.join(images_path, label_dir)

        for file_name in sorted(os.listdir(label_path)):
            file_path = os.path.join(label_path, file_name)
            
            img_paths.append(file_path)
            img_labels.append(label)
        i += 1

    # Performing split of dataset
    # First splitting test set and the rest
    TEST_PORTION = 3000/len(img_paths) # splits approx. 3000 of features to test set
    img_paths_temp, img_paths_test, img_labels_temp, img_labels_test = train_test_split(img_paths, img_labels, test_size=TEST_PORTION, stratify=img_labels, random_state=seed)

    # Then splitting training and validation sets
    VAL_PORTION = 2000/len(img_paths_temp) # approx. 2000 features in validation set
    img_paths_train, img_paths_val, img_labels_train, img_labels_val = train_test_split(img_paths_temp, img_labels_temp, test_size=VAL_PORTION, stratify=img_labels_temp, random_state=seed)

    # Asserting that sets are disjoint
    assert set(img_paths_train).isdisjoint(set(img_paths_val))
    assert set(img_paths_train).isdisjoint(set(img_paths_test))
    assert set(img_paths_val).isdisjoint(set(img_paths_test))

    # Creating Datasets and DataLoaders
    train_dataset = ImageDataset(img_paths_train, img_labels_train)
    val_dataset = ImageDataset(img_paths_val, img_labels_val)
    test_dataset = ImageDataset(img_paths_test, img_labels_test)

    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=True)

    return train_dataset, val_dataset, test_dataset, train_dataloader, val_dataloader, test_dataloader