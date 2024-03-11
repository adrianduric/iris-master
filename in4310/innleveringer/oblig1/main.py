from data_mgmt import *
from model import *
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


# Setting random seed for testing
RANDOM_SEED = 77
torch.manual_seed(RANDOM_SEED)

# Setting hyperparameters
config = {
          'batch_size': 16,
          'use_cuda': True,
          'epochs': 10,
          'learningRate': 1e-3
         }

# Creating DataLoaders from images
train_dataloader, val_dataloader, test_dataloader = create_dataloaders(config)

# Storing data from runs
accuracies = []
ap_scores = []
mean_ap_scores = []
    
# Training and evaluating 3 models
for model_num in range(3):

    # Task 1c)
    # Loading ResNet18 pretrained with weights
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    # Swapping last layer for classification of 6 labels
    model.fc = nn.Linear(in_features=512, out_features=6, bias=True)

    if config["use_cuda"]:
        model.to("cuda")

    # Choosing CrossEntropyLoss as loss function
    loss_fn = nn.CrossEntropyLoss()

    # Choosing Adam as optimizer
    optimizer = torch.optim.SGD(model.parameters())
    # Train model for specified amount of epochs
    for e in range(config["epochs"]):
        print(f"----------- EPOCH {e+1} -----------")
        train_model(train_dataloader, model, loss_fn, optimizer, config)

        # Tracking metrics on validation sets during training
        accuracy, ap_score, mean_ap_score = test_model(model, val_dataloader)
        accuracies.append(accuracy)
        ap_scores.append(ap_score)
        mean_ap_scores.append(mean_ap_score)

    plt.plot(range(config["epochs"]), accuracies)
    plt.plot(range(config["epochs"]), ap_scores)
    plt.plot(range(config["epochs"]), mean_ap_scores)
    plt.show()
    