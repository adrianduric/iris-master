import torch
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from getimagenetclasses import parseclasslabel, parsesynsetwords, get_classes
# Try other models https://pytorch.org/vision/stable/models.html
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import accuracy_score


class ImageNet2500(Dataset):
    def __init__(self, root_dir, xmllabeldir, synsetfile, images_dir, transform=None):

        """
    Args:

        root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied
            on a sample.
    """

        self.root_dir = root_dir
        self.xmllabeldir = os.path.join(root_dir, xmllabeldir)
        self.images_dir = os.path.join(root_dir, images_dir)
        self.transform = transform
        self.imgfilenames = []
        self.labels = []
        self.ending = ".JPEG"

        indicestosynsets, self.synsetstoindices, synsetstoclassdescr = parsesynsetwords(os.path.join(root_dir, synsetfile))

        for file in os.listdir(self.images_dir):
            if file.endswith(".JPEG"):
                name = os.path.join(self.images_dir, file)
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
        img = Image.open(self.imgfilenames[idx])
        if img.mode != "RGB":
            img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

def run_model(model, dataloader):
    pred = torch.Tensor()
    lbls = torch.Tensor()

    for batch_idx, data in enumerate(dataloader):
        prediction = model(data[0])
        label = data[1]
        torch.cat((pred, prediction), 0)
        torch.cat((lbls, label), 0)

    return pred, lbls


def plot_example(indx, model, dataset):
    sample = dataset[indx]
    plt.imshow(sample[0].permute(1, 2, 0))
    plt.show()
    # im = transforms.ToPILImage()(sample["image"])
    # im.show()
    prediction = model(sample[0].unsqueeze(0)).detach().numpy()[0]
    ind = prediction.argsort()[-5:][::-1]
    print("Top-5 predicted levels:\n")
    for key in ind:
        print(get_classes().get(key))

    print("\nTrue label ", get_classes()[sample[1]])


def compare_performance(model, loader_wo_normalize, loader_w_normalize):
    # predictions and labels from dataset without normalization
    preds, labels = run_model(model, loader_wo_normalize)
    # predictions and labels from dataset with normalization (labels are the same as before)
    preds_norm, _ = run_model(model, loader_w_normalize)

    acc = accuracy_score(labels, preds)
    acc_norm = accuracy_score(preds_norm, labels)

    print("Accuracy without normalize: ", acc)
    print("Accuracy with normalize: ", acc_norm)


if __name__ == "__main__":
    main_path = "/home/adrian/iris-master/in4310/ukesoppgaver/uke4"
    # These files/folders should be inside the main_path directory, i.e.
    # ../solution /
    # ├── ILSVRC2012_bbox_val_v3 /
    # │   └── val /
    # ├── imagenet2500 /
    # │   └── imagespart /
    # ├── getimagenetclasses.py
    # └── synset_words.txt

    xmllabeldir = "ILSVRC2012_bbox_val_v3/val/"
    synsetfile = 'synset_words.txt'
    images_dir = "imagenet2500/imagespart"

    #  https://pytorch.org/vision/main/generated/torchvision.transforms.Compose.html
    base_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    normalize_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset_wo_normalize = ImageNet2500(root_dir=main_path, xmllabeldir=xmllabeldir, synsetfile=synsetfile, images_dir=images_dir, transform=base_transform)
    loader_wo_normalize = DataLoader(dataset_wo_normalize, batch_size=64, shuffle=True)

    dataset_w_normalize = ImageNet2500(root_dir=main_path, xmllabeldir=xmllabeldir, synsetfile=synsetfile, images_dir=images_dir, transform=normalize_transform)
    loader_w_normalize = DataLoader(dataset_w_normalize, batch_size=64, shuffle=True)

    # load a pretrained model of choice (see models in torchvision.models)
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    # Set model to eval mode to use the learned-statistics instead of batch-statistics for batch_norm, and skip
    # training-only operations like dropout. Try removing this line and see how the model performs!
    model.eval()

    compare_performance(model, loader_wo_normalize, loader_w_normalize)
    # change the index to check other examples
    plot_example(6, model, dataset_wo_normalize)
