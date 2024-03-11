import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from tqdm import tqdm


# Task 1e)
# Train model for one epoch
def train_model(dataloader, model, loss_fn, optimizer, config):

    # Iterating through batches
    for batch_idx, (batch_images, batch_labels) in enumerate(tqdm(dataloader)):
        if config["use_cuda"]:
            batch_images = batch_images.to("cuda")
            batch_labels = batch_labels.to("cuda")
            
        # Forward pass and loss calculation
        batch_predictions = model(batch_images)
        loss = loss_fn(batch_predictions, batch_labels)
        
        # Backpropagation and updating weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
#end train_model

# Test model on testing or validation data
def test_model(model, dataloader, config):

    all_predictions = torch.Tensor()
    all_labels = torch.Tensor()
    all_labels_one_hot = torch.Tensor()
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
            batch_predictions = model(batch_images)
            batch_labels_one_hot = nn.functional.one_hot(batch_labels.long(), num_classes=6)

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
        
    return accuracy, ap_score, mean_ap_score
#end test_model













