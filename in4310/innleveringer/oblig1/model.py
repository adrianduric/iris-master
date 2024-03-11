import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
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
        batch_predictions = model(batch_images)
        loss = loss_fn(batch_predictions, batch_labels)
        total_loss += loss.item()
        
        # Backpropagation and updating weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return total_loss


# Test model on testing or validation data
def test_model(dataloader, model, loss_fn, config):

    all_predictions = torch.Tensor()
    all_labels = torch.Tensor()
    all_labels_one_hot = torch.Tensor()
    total_loss = 0

    if config["use_cuda"]:
        all_predictions = all_predictions.to("cuda")
        all_labels = all_labels.to("cuda")
        all_labels_one_hot = all_labels_one_hot.to("cuda")

    # Iterating through batches
    for batch_idx, (batch_images, batch_labels) in enumerate(tqdm(dataloader)):
        if config["use_cuda"]:
            batch_images = batch_images.to("cuda")
            batch_labels = batch_labels.to("cuda")
        with torch.no_grad():
            batch_labels_one_hot = nn.functional.one_hot(batch_labels.long(), num_classes=6)
            batch_predictions = model(batch_images)
            loss = loss_fn(batch_predictions, batch_labels)
            total_loss += loss.item()

            all_predictions = torch.cat((all_predictions, batch_predictions), 0)
            all_labels = torch.cat((all_labels, batch_labels), 0)
            all_labels_one_hot = torch.cat((all_labels_one_hot, batch_labels_one_hot), 0)

    # Task 1d)
    # Calculating performance metrics
    all_predictions = all_predictions.to("cpu")
    all_labels = all_labels.to("cpu")
    all_labels_one_hot = all_labels_one_hot.to("cpu")

    accuracy = torch.sum(torch.argmax(all_predictions, dim=1) == all_labels) / len(all_labels)
    ap_score = average_precision_score(all_labels_one_hot, all_predictions, average=None)
    mean_ap_score = average_precision_score(all_labels_one_hot, all_predictions, average="macro")
        
    return total_loss, accuracy, ap_score, mean_ap_score













