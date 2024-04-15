from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

from utils.cv2_augmentor import CV2Augmentor
from utils.stain_separation import separate_stain


class CONSEP(Dataset):
    def __init__(self, root_dir, mode, transform=None):
        """root_dir should contain directories `tiled_images` and `tiled_labels`"""
        super(CONSEP, self).__init__()

        self.root_dir = Path(root_dir)
        self.mode = mode
        self.transform = transform
        self.file_names = [i.stem for i in (self.root_dir / 'tiled_images').glob('*.png')]
        self.augmentor = CV2Augmentor((270, 270), mode, 0)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        # Load image
        img_path = self.root_dir / 'tiled_images' / f'{self.file_names[idx]}.png'
        x = Image.open(img_path).convert('RGB')
        # Load labels
        labels_path = self.root_dir / 'tiled_labels' / f'{self.file_names[idx]}.npy'
        y = np.load(str(labels_path))

        x = cv2.cvtColor(np.array(x), cv2.COLOR_RGB2BGR)  # Convert image to an OpenCV image for augmentor
        x, y = self.augmentor(x, y)  # Augment the image using the augmentations used in HoverNet

        # Extract H-stain
        h_x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        h_x = separate_stain(h_x)[:, :, 0]
        h_x = to_tensor(h_x)

        if self.transform is not None:
            x = self.transform(x)
        else:
            x = to_tensor(x)

        y = torch.FloatTensor(y.copy())
        return x, h_x, y
