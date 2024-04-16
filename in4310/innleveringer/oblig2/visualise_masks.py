from pathlib import Path

import torch
from torchvision.models import resnet18
from torchvision.utils import save_image

from resnet_unet import TwoEncodersOneDecoder


def visualise_segmentation(model_path, destination_path, dataloader):
    """
    Visualises the output of a model as an image with 3 columns:
    1st column is the original input image to be segmented.
    2nd column is the output of the model -> the segmentation mask.
    3rd column is the ground truth segmentation mask.

    :param model_path: The path to the model checkpoint file
    :param destination_path: The path to the directory where you'd like to save the images
    :param dataloader: A pytorch dataloader that provides the input images to be segmented
    """
    model = TwoEncodersOneDecoder(resnet18, pretrained=False, out_channels=1)
    model.load_state_dict(torch.load(model_path)['model'])
    model.eval()
    cuda_device = torch.device('cuda', 0)
    model.to(cuda_device)  # Note: comment out this line if you'd like to run the model on the CPU locally

    destination_path = Path(destination_path)
    destination_path.mkdir(exist_ok=True)
    for batch_idx, (x, h_x, y) in enumerate(dataloader):
        x.to(cuda_device)
        h_x.to(cuda_device)
        y.to(cuda_device)

        h_x.expand(-1, 3, -1, -1)

        with torch.no_grad():
            out = model(x, h_x)

            probs = torch.sigmoid(out)
            mask = torch.where(x > 0.5, 1, 0)

        edges = []
        # TODO: Convert each image in the batch "out" to have 3 channels and store them in the list "edges"
        out.expand(-1, 3, -1, -1)
        edges.append(out)

        assert len(edges) == x.size(0), f'Expected {x.size(0)} elements in edges but got {len(edges)} instead.'

        for i in range(len(edges)):
            img_name = str(batch_idx * dataloader.batch_size + i)
            gt = y[i].expand(-1, 3, -1, -1)
            image = torch.stack([x[i], edges[i], gt])
            save_image(image.float().to('cpu'),
                       destination_path / f'{img_name}.jpg',
                       padding=10, pad_value=0.5)
