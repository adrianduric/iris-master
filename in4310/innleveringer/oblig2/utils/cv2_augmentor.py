from imgaug import augmenters as iaa
from utils.hovernet_augs import add_to_brightness, add_to_contrast, add_to_hue, add_to_saturation, gaussian_blur, median_blur


class CV2Augmentor:
    def __init__(self, input_shape, mode, seed):
        self.input_shape = input_shape
        self.mode = mode
        self.augmentor = self._get_augmentation(seed)
        self.shape_augs = iaa.Sequential(self.augmentor[0])
        self.input_augs = iaa.Sequential(self.augmentor[1])

    def _get_augmentation(self, seed):
        if self.mode == 'train':
            shape_augs = [
                # * order = ``0`` -> ``cv2.INTER_NEAREST``
                # * order = ``1`` -> ``cv2.INTER_LINEAR``
                # * order = ``2`` -> ``cv2.INTER_CUBIC``
                # * order = ``3`` -> ``cv2.INTER_CUBIC``
                # * order = ``4`` -> ``cv2.INTER_CUBIC``
                # ! for pannuke v0, no rotation or translation, just flip to avoid mirror padding
                iaa.Affine(
                    # scale images to 80-120% of their size, individually per axis
                    scale={'x': (0.8, 1.2), 'y': (0.8, 1.2)},
                    # translate by -A to +A percent (per axis)
                    translate_percent={'x': (-0.01, 0.01), 'y': (-0.01, 0.01)},
                    shear=(-5, 5),  # shear by -5 to +5 degrees
                    rotate=(-179, 179),  # rotate by -179 to +179 degrees
                    order=0,  # use nearest neighbour
                    backend='cv2',  # opencv for fast processing
                    seed=seed,
                ),
                # set position to 'center' for center crop
                # else 'uniform' for random crop
                iaa.CropToFixedSize(
                    self.input_shape[0], self.input_shape[1], position="center"
                ),
                iaa.Fliplr(0.5, seed=seed),
                iaa.Flipud(0.5, seed=seed),
            ]

            input_augs = [
                iaa.OneOf(
                    [
                        iaa.Lambda(
                            seed=seed,
                            func_images=lambda *args: gaussian_blur(*args, max_ksize=3),
                        ),
                        iaa.Lambda(
                            seed=seed,
                            func_images=lambda *args: median_blur(*args, max_ksize=3),
                        ),
                        iaa.AdditiveGaussianNoise(
                            loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                        ),
                    ]
                ),
                iaa.Sequential(
                    [
                        iaa.Lambda(
                            seed=seed,
                            func_images=lambda *args: add_to_hue(*args, range=(-8, 8)),
                        ),
                        iaa.Lambda(
                            seed=seed,
                            func_images=lambda *args: add_to_saturation(
                                *args, range=(-0.2, 0.2)
                            ),
                        ),
                        iaa.Lambda(
                            seed=seed,
                            func_images=lambda *args: add_to_brightness(
                                *args, range=(-26, 26)
                            ),
                        ),
                        iaa.Lambda(
                            seed=seed,
                            func_images=lambda *args: add_to_contrast(
                                *args, range=(0.75, 1.25)
                            ),
                        ),
                    ],
                    random_order=True,
                ),
            ]
        elif self.mode == 'val':
            shape_augs = [
                # set position to 'center' for center crop
                # else 'uniform' for random crop
                iaa.CropToFixedSize(
                    self.input_shape[0], self.input_shape[1], position="center"
                )
            ]
            input_augs = []

        return shape_augs, input_augs

    def __call__(self, x, y=None):
        if self.shape_augs is not None:
            shape_augs = self.shape_augs.to_deterministic()
            x = shape_augs.augment_image(x)
            if y is not None:
                y = shape_augs.augment_image(y)
        if self.input_augs is not None:
            input_augs = self.input_augs.to_deterministic()
            x = input_augs.augment_image(x)
        if y is None:
            return x
        return x, y
