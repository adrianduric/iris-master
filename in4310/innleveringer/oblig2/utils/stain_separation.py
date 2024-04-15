import numpy as np


def separate_stain(img):
    H = np.array([0.650, 0.704, 0.286])
    E = np.array([0.072, 0.990, 0.105])
    R = np.array([0.268, 0.570, 0.776])

    HDABtoRGB = [(H/np.linalg.norm(H)).tolist(), (E/np.linalg.norm(E)).tolist(), (R/np.linalg.norm(R)).tolist()]
    stain_matrix = HDABtoRGB
    im_inv = np.linalg.inv(stain_matrix)
    im_temp = (-255) * np.log((np.float64(img) + 1) / 255) / np.log(255)
    image_out = np.reshape(np.dot(np.reshape(im_temp, [-1, 3]), im_inv), np.shape(img))
    image_out = np.exp((255-image_out) * np.log(255) / 255)
    image_out[image_out > 255] = 255

    return np.uint8(image_out)


if __name__ == '__main__':
    from pathlib import Path
    import cv2
    import skimage

    destination = Path('H_stain')
    destination.mkdir(parents=True, exist_ok=True)

    for i in range(100):
        # img = Image.open(f'tiles/{i}.jpg').convert('RGB')
        img = cv2.imread(f'tiles/{i}.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        he = separate_stain(img)[:, :, 0]
        he = skimage.color.gray2rgb(he)
        skimage.io.imsave(destination / f'{i}.jpg', he)
        # im = Image.fromarray(img)
        # im.save(destination / f'{i}.jpg')
