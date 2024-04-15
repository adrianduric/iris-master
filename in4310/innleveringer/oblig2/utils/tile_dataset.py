from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat


def tile_images_dir(root_dir, dst_dir, tile_size=(224, 224), step_size=70, pad=False,
                    create_tile_for_remaining_pixels=False):
    """Expects the root_dir to have .png files."""
    root_dir = Path(root_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in root_dir.glob('*.png'):
        img = np.array(Image.open(root_dir / f), dtype=np.uint8)
        tiles = tile_image(img, tile_size, step_size, pad, create_tile_for_remaining_pixels)
        # Save the tiles
        for i, t in enumerate(tiles):
            tile_pil = Image.fromarray(t.astype(np.uint8))
            tile_pil.save(dst_dir / f'{f.stem}_{i}.png')


def tile_labels_dir(root_dir, dst_dir, tile_size=(224, 224), step_size=70, pad=False,
                    create_tile_for_remaining_pixels=False):
    """Expects the root_dir to have .mat files."""
    root_dir = Path(root_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in root_dir.glob('*.mat'):
        data = loadmat(str(root_dir / f))
        # Convert 1-d labels to a 3-d image by copying labels for each channel
        # This conversion allows us to use out tile_image function
        labels = np.repeat(data['inst_map'][:, :, np.newaxis], 3, axis=2)
        # We don't care about nuclei types so just set all the non-zero labels to 1
        labels[labels > 0] = 1
        labels = labels.astype(np.ushort)
        tiles = tile_image(labels, tile_size, step_size, pad, create_tile_for_remaining_pixels)
        tiles = [t[..., 0].astype(np.uint8) for t in tiles]
        # Save the tile labels
        for i, t in enumerate(tiles):
            np.save(str(dst_dir / f'{f.stem}_{i}.npy'), t)


def tile_image(img, tile_size, step_size, pad=False, create_tile_for_remaining_pixels=False):
    """
    Tile the image with, potentially with reflect(mirror) padding on all sides to not leave out any pixels.

    :param img: An image as a numpy array
    :param tile_size: A tuple (height, width) denoting the desired size of tiles
    :param step_size: The number of pixels to skip (in both directions) to start the next tile
    :param pad: Pad the image with reflect(mirror) padding if True
    :param create_tile_for_remaining_pixels: If there are more than 30 pixels left out towards the end then create
                                             an extra tile to cover those pixels even though it leads to overlap.

    :return: A list containing all tiles as numpy arrays
    """
    original_height, original_width, _ = img.shape
    padded_image = img

    if pad:
        # Calculate the number of rows and columns for the padded image
        num_rows = (original_height + tile_size[0] - 1) // tile_size[0]
        num_cols = (original_width + tile_size[1] - 1) // tile_size[1]

        # Calculate the padding required for both sides
        pad_height = num_rows * tile_size[0] - original_height
        pad_width = num_cols * tile_size[1] - original_width

        # Pad the original image with mirrored borders
        padded_image = np.pad(img, ((pad_height // 2, pad_height - pad_height // 2),
                                    (pad_width // 2, pad_width - pad_width // 2),
                                    (0, 0)), mode='reflect')
    tiles = []
    for y in range(0, padded_image.shape[0], step_size):
        for x in range(0, padded_image.shape[1], step_size):
            start_x = x
            start_y = y

            end_y = y + tile_size[0]
            end_x = x + tile_size[1]

            if end_x > padded_image.shape[1]:
                if create_tile_for_remaining_pixels:
                    start_x = padded_image.shape[1] - tile_size[1]
                    end_x = padded_image.shape[1]
                else:
                    continue

            if end_y > padded_image.shape[0]:
                if create_tile_for_remaining_pixels:
                    start_y = padded_image.shape[0] - tile_size[0]
                    end_y = padded_image.shape[0]
                else:
                    continue

            sub_image = padded_image[start_y:end_y, start_x:end_x, :]
            tiles.append(sub_image)

    return tiles


if __name__ == '__main__':
    # Train subset
    tile_images_dir('/work/consep_nuclei_segmentation_data/train/Images',
                    '/work/consep_nuclei_segmentation_data/train/tiled_images',
                    tile_size=(224, 224), step_size=70, pad=False)
    tile_labels_dir('/work/consep_nuclei_segmentation_data/train/Labels',
                    '/work/consep_nuclei_segmentation_data/train/tiled_labels',
                    tile_size=(224, 224), step_size=70, pad=False)
    # Validation subset
    tile_images_dir('/work/consep_nuclei_segmentation_data/val/Images',
                    '/work/consep_nuclei_segmentation_data/val/tiled_images',
                    tile_size=(224, 224), step_size=224, pad=False, create_tile_for_remaining_pixels=True)
    tile_labels_dir('/work/consep_nuclei_segmentation_data/val/Labels',
                    '/work/consep_nuclei_segmentation_data/val/tiled_labels',
                    tile_size=(224, 224), step_size=224, pad=False, create_tile_for_remaining_pixels=True)
    # Test subset
    tile_images_dir('/work/consep_nuclei_segmentation_data/test/Images',
                    '/work/consep_nuclei_segmentation_data/test/tiled_images',
                    tile_size=(224, 224), step_size=224, pad=False, create_tile_for_remaining_pixels=True)
    tile_labels_dir('/work/consep_nuclei_segmentation_data/test/Labels',
                    '/work/consep_nuclei_segmentation_data/test/tiled_labels',
                    tile_size=(224, 224), step_size=224, pad=False, create_tile_for_remaining_pixels=True)
