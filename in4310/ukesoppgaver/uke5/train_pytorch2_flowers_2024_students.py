from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torchvision.models import resnet18
#import matplotlib.pyplot as plt
import time
import os

from tqdm import tqdm

#import skimage.io
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils



class dataset_flowers(Dataset):
  def __init__(self, root_dir, trvaltest, transform=None):


    self.root_dir = root_dir
    self.images_dir = os.path.join(self.root_dir, "flowers_data/jpg")

    self.transform = transform
    self.imgfilenames=[]
    self.labels=[]
    

    if trvaltest==0:
      # load training data
      file = open("flowers_data/trainfile.txt", "r")
      for line in file:
        words = line.split()
        imgfilename = os.path.join(self.images_dir, words[0])
        label = int(words[1])
        self.imgfilenames.append(imgfilename)
        self.labels.append(label)
      file.close()

    elif trvaltest==1:
      # load validation data
      file = open("flowers_data/valfile.txt", "r")
      for line in file:
        words = line.split()
        imgfilename = os.path.join(self.images_dir, words[0])
        label = int(words[1])
        self.imgfilenames.append(imgfilename)
        self.labels.append(label)
      file.close()

    elif trvaltest==2:
      # load testing data
      file = open("flowers_data/valfile.txt", "r")
      for line in file:
        words = line.split()
        imgfilename = os.path.join(self.images_dir, words[0])
        label = int(words[1])
        self.imgfilenames.append(imgfilename)
        self.labels.append(label)
      file.close()

    else:
      raise Exception(f"Invalid input for trvaltest (expected 0,1,2, got {trvaltest}).")       

  def __len__(self):
      return len(self.labels)

  def __getitem__(self, idx):

    filename = self.imgfilenames[idx]
    label = self.labels[idx]
    image = Image.open(filename)
    if self.transform:
      image = self.transform(image)
    sample = {'image': image, 'label': label, 'filename': filename}

    return sample


def train_epoch(model,  trainloader,  losscriterion, device, optimizer ):

    model.train() # IMPORTANT
 
    losses = list()
    for batch_idx, data in enumerate(tqdm(trainloader)):
      imgs = data["image"].to(device)
      lbls = data["label"].to(device)

      # Forward pass
      preds = model(imgs)
      loss = losscriterion(preds, lbls)
      losses.append(loss.item())

      # Backprop
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

    return np.mean(losses)


def evaluate_acc(model, dataloader, losscriterion, device):

    model.eval() # IMPORTANT

    losses = []
    curcount = 0
    accuracy = 0
    
    with torch.no_grad():
      for ctr, data in enumerate(tqdm(dataloader)):
        imgs = data["image"].to(device)
        labels = data["label"].to(device)
        #computes predictions on samples from the dataloader
        preds = model(imgs)
        loss = losscriterion(preds, labels)
        losses.append(loss.item())

        # computes accuracy, to count how many samples, you can just sum up labels.shape[0]
        corrects = torch.sum(torch.argmax(preds, dim=1) == labels) / float(labels.shape[0])
        accuracy = accuracy * (curcount / float(curcount + labels.shape[0])) + corrects.float() * (
                    labels.shape[0] / float(curcount + labels.shape[0]))
        curcount += labels.shape[0]

    return accuracy.item() , np.mean(losses)


def train_model_nocv_sizes(dataloader_train, dataloader_test ,  model ,  losscriterion, optimizer, scheduler, num_epochs, device):

  best_measure = 0
  best_epoch =-1
  bestweights = model.state_dict()

  for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    model.train(True)
    losses=train_epoch(model,  dataloader_train,  losscriterion,  device , optimizer )

    if scheduler is not None:
      scheduler.step()

    model.train(False)
    measure, meanlosses = evaluate_acc(model, dataloader_test, losscriterion, device)
    print(' perfmeasure', measure)

    if measure > best_measure:
      # save the weights of the best model
      bestweights = model.state_dict()
      # update   best_measure, best_epoch
      best_measure = measure
      best_epoch = epoch

  return best_epoch, best_measure, bestweights



def runstuff_finetunealllayers():

  #someparameters
  batchsize_tr=16
  batchsize_test=16
  maxnumepochs=2 

  device= torch.device('cuda')

  numcl=102
  #transforms
  data_transforms = {}
  data_transforms['train']=transforms.Compose([
          transforms.Resize(224),
          transforms.RandomCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ])
  data_transforms['val']=transforms.Compose([
          transforms.Resize(224),
          transforms.CenterCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ])
  data_transforms['test']=transforms.Compose([
          transforms.Resize(224),
          transforms.CenterCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ])
  
  datasets={}
  datasets['train'] = dataset_flowers(root_dir="/home/adrian/iris-master/in4310/ukesoppgaver/uke5", trvaltest=0, transform=data_transforms["train"])
  datasets['val'] = dataset_flowers(root_dir="/home/adrian/iris-master/in4310/ukesoppgaver/uke5", trvaltest=1, transform=data_transforms["val"])
  datasets['test'] = dataset_flowers(root_dir="/home/adrian/iris-master/in4310/ukesoppgaver/uke5", trvaltest=2, transform=data_transforms["test"])

  dataloaders={}
  dataloaders['train'] = DataLoader(datasets["train"], shuffle=True)
  dataloaders['val'] = DataLoader(datasets["val"], shuffle=True)
  dataloaders['test'] = DataLoader(datasets["test"], shuffle=True)

  #model
  model = resnet18()

  model.to(device)

  criterion = nn.CrossEntropyLoss()

  lrates=[0.01, 0.001]

  best_hyperparameter= None
  weights_chosen = None
  bestmeasure = None
  for lr in lrates:

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_epoch, best_perfmeasure, bestweights = train_model_nocv_sizes(dataloader_train = dataloaders['train'], dataloader_test = dataloaders['val'] ,  model = model ,  losscriterion = criterion , optimizer = optimizer, scheduler = None, num_epochs = maxnumepochs , device = device)

    if best_hyperparameter is None:
      best_hyperparameter = lr
      weights_chosen = bestweights
      bestmeasure = best_perfmeasure
     
    elif best_perfmeasure > bestmeasure:
      best_hyperparameter = lr
      weights_chosen = bestweights
      bestmeasure = best_perfmeasure

  model.load_state_dict(weights_chosen)

  accuracy, testloss = evaluate_acc(model = model , dataloader  = dataloaders['test'], losscriterion = criterion, device = device)

  print('accuracy val',bestmeasure , 'accuracy test',accuracy  )



def runstuff_finetunelastlayer():

  pass
  #TODO 


def runstuff_fromscratch():

  pass
  #TODO 




if __name__=='__main__':

  #runstuff_fromscratch()
  runstuff_finetunealllayers()
  #runstuff_finetunelastlayer()



