import torch
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from getimagenetclasses import parseclasslabel, parsesynsetwords, get_classes
# Try other models https://pytorch.org/vision/stable/models.html
from torchvision.models import resnet18, ResNet18_Weights


class ImageNet2500(Dataset):
    def __init__(self, root_dir, xmllabeldir, synsetfile, images_dir, transform=None):

        """
    Args:

        root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied
            on a sample.
    """

        self.root_dir = root_dir
        self.xmllabeldir = root_dir + xmllabeldir
        self.images_dir = root_dir + images_dir
        self.transform = transform
        self.imgfilenames = []
        self.labels = []
        self.ending = ".JPEG"

        indicestosynsets, self.synsetstoindices, synsetstoclassdescr = parsesynsetwords(root_dir + synsetfile)

        for file in os.listdir(self.images_dir):
            if file.endswith(".JPEG"):
                name = os.path.join(images_dir, file)
                self.imgfilenames.append(name)
                label, _ = parseclasslabel(self.filenametoxml(name), self.synsetstoindices)
                self.labels.append(label)

    def filenametoxml(self, fn):
        f = os.path.basename(fn)

        if not f.endswith(self.ending):
            print('not f.endswith(self.ending)')
            exit()

        f = f[:-len(self.ending)] + '.xml'
        f = os.path.join(self.xmllabeldir, f)

        return f

    def __len__(self):
        return len(self.imgfilenames)

    def __getitem__(self, idx):
        image = Image.open(self.imgfilenames[idx]).convert('RGB')

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        sample = {'image': image, 'label': label, 'filename': self.imgfilenames[idx]}

        return sample

def run_model(model, dataloader):
    pred = torch.Tensor()
    lbls = torch.Tensor()

    for batch_idx, data in enumerate(dataloader):
        prediction = model(data["image"])
        pred = torch.cat((pred, prediction), 0)
        lbls = torch.cat((lbls, data["label"]), 0)
        if batch_idx == 8:
            break

    return pred, lbls


def plot_example(indx, model, dataset):
    sample = dataset[indx]
    plt.imshow(sample["image"].permute(1, 2, 0))
    plt.show()
    # im = transforms.ToPILImage()(sample["image"])
    # im.show()
    prediction = model(sample["image"].unsqueeze(0)).detach().numpy()[0]
    ind = prediction.argsort()[-5:][::-1]
    print("Top-5 predicted levels:\n")
    for key in ind:
        print(get_classes().get(key))

    print("\nTrue label ", get_classes()[sample["label"]])


def compare_performance(model, loader_wo_normalize, loader_w_normalize):
    # predictions and labels from dataset without normalization
    preds, labels = run_model(model, loader_wo_normalize)
    # predictions and labels from dataset with normalization (labels are the same as before)
    preds_norm, _ = run_model(model, loader_w_normalize)

    acc = torch.sum(torch.argmax(preds, dim=1) == labels) / len(labels)
    acc_norm = torch.sum(torch.argmax(preds_norm, dim=1) == labels) / len(labels)

    print("Accuracy without normalize: ", acc)
    print("Accuracy with normalize: ", acc_norm)


if __name__ == "__main__":
    main_path = "/path/to/files/"
    # These files/folders should be inside the main_path directory, i.e.
    # ..Solution /
    # ├── ILSVRC2012_bbox_val_v3 /
    # │   └── val /
    # ├── imagenet2500 /
    # │   └── imagespart /
    # ├── getimagenetclasses.py
    # └── synset_words.txt

    xmllabeldir = "/ILSVRC2012_bbox_val_v3/val/"
    synsetfile = '/synset_words.txt'
    images_dir = "imagenet2500/imagespart"

    base_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    normalize_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset_wo_normalize = ImageNet2500(main_path, xmllabeldir, synsetfile, images_dir, transform=base_transform)
    loader_wo_normalize = DataLoader(dataset_wo_normalize, batch_size=64, shuffle=False)

    dataset_w_normalize = ImageNet2500(main_path, xmllabeldir, synsetfile, images_dir, transform=normalize_transform)
    loader_w_normalize = DataLoader(dataset_w_normalize, batch_size=64, shuffle=False)

    # load a pretrained resnet18 model
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1) # todo: try to use other models from the torchvision model zoo
    # Set model to eval mode to use the learned-statistics instead of batch-statistics for batch_norm, and skip
    # training-only operations like dropout. Try removing this line and see how the model performs!
    model.eval()

    compare_performance(model, loader_wo_normalize, loader_w_normalize)
    # change the index to check other examples
    plot_example(6, model, dataset_wo_normalize)
