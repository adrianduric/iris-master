import time
from datetime import timedelta

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision.transforms import transforms

from datasets import CONSEP
from resnet_unet import TwoEncodersOneDecoder
from utils.plotting import plot_loss

import os
from tqdm import tqdm

from train import save_checkpoint, dice_loss_fn, eval_dice_with_h_x
from visualise_masks import visualise_segmentation

cuda_device = torch.device('cuda', 0)

def test():
    model = TwoEncodersOneDecoder(resnet18, pretrained=True, out_channels=1)
    model.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "TwoEncodersOneDecoder_consep.pth"))["model"])
    model.eval()
    model.to(cuda_device)

    dataset_test = CONSEP(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/test"), mode='val')
    dataloader_test = DataLoader(dataset_test, batch_size=32, num_workers=10, pin_memory=True)

    print('EVALUATING dice score on test set')
    dice_score_test = eval_dice_with_h_x(model, dataloader_test)

    visualise_segmentation(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "TwoEncodersOneDecoder_consep.pth"),
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "results"),
        dataloader_test
        )

if __name__ == '__main__':
    test()
