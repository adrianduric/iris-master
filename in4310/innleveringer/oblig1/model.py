import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, confusion_matrix
from tqdm import tqdm


# Task 1e)
# Train model for one epoch
def train_model(dataloader, model, loss_fn, optimizer, config):

    total_loss = 0

    # Iterating through batches
    for batch_idx, (batch_images, batch_labels) in enumerate(tqdm(dataloader)):
        if config["use_cuda"]:
            batch_images = batch_images.to("cuda")
            batch_labels = batch_labels.to("cuda")
            
        # Forward pass and loss calculation
        batch_softmaxes = model(batch_images)
        loss = loss_fn(batch_softmaxes, batch_labels)
        total_loss += loss.item()
        
        # Backpropagation and updating weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return total_loss


# Test model on testing or validation data
def test_model(dataloader, model, loss_fn, config, get_softmax=False):

    all_softmaxes = torch.Tensor()
    all_predictions = torch.Tensor()
    all_labels = torch.Tensor()
    all_labels_one_hot = torch.Tensor()
    total_loss = 0

    if config["use_cuda"]:
        all_predictions = all_predictions.to("cuda")
        all_softmaxes = all_softmaxes.to("cuda")
        all_labels = all_labels.to("cuda")
        all_labels_one_hot = all_labels_one_hot.to("cuda")

    # Iterating through batches
    for batch_idx, (batch_images, batch_labels) in enumerate(tqdm(dataloader)):
        if config["use_cuda"]:
            batch_images = batch_images.to("cuda")
            batch_labels = batch_labels.to("cuda")
        with torch.no_grad():
            
            batch_softmaxes = model(batch_images)
            loss = loss_fn(batch_softmaxes, batch_labels)
            total_loss += loss.item()

            batch_labels_one_hot = nn.functional.one_hot(batch_labels.long(), num_classes=6)
            batch_predictions = torch.argmax(batch_softmaxes, dim=1)
            
            all_predictions = torch.cat((all_predictions, batch_predictions), 0)
            all_softmaxes = torch.cat((all_softmaxes, batch_softmaxes), 0)
            all_labels = torch.cat((all_labels, batch_labels), 0)
            all_labels_one_hot = torch.cat((all_labels_one_hot, batch_labels_one_hot), 0)

    # Task 1d)
    # Calculating performance metrics
    all_predictions = all_predictions.to("cpu")
    all_softmaxes = all_softmaxes.to("cpu")
    all_labels = all_labels.to("cpu")
    all_labels_one_hot = all_labels_one_hot.to("cpu")

    conf_matrix = confusion_matrix(all_labels, all_predictions)
    accuracy_per_class = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    mean_accuracy_per_class = np.mean(accuracy_per_class)
    ap_score = average_precision_score(all_labels_one_hot, all_softmaxes, average=None) # Part of task 1d)
    mean_ap_score = average_precision_score(all_labels_one_hot, all_softmaxes, average="macro")

    if get_softmax:
        return all_softmaxes, total_loss, accuracy_per_class, mean_accuracy_per_class, ap_score, mean_ap_score
    return total_loss, accuracy_per_class, mean_accuracy_per_class, ap_score, mean_ap_score













