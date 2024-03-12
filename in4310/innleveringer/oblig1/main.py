from data_mgmt import *
from model import *
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models import resnet18


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
models = []
training_losses = []
val_losses = []
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

    model_training_losses = []
    model_val_losses = []
    model_accuracies = []
    model_ap_scores = []
    model_mean_ap_scores = []


    # Train model for specified amount of epochs
    for e in range(config["epochs"]):
        print(f"----------- EPOCH {e+1} -----------")
        print("Training model...")
        epoch_training_loss = train_model(train_dataloader, model, loss_fn, optimizer, config)
        print("Training complete.\n")

        # Tracking metrics on validation sets during training
        print("Testing model on validation set...")
        epoch_val_loss, epoch_acc, epoch_ap, epoch_mean_ap = test_model(val_dataloader, model, loss_fn, config)
        print("Testing complete.\n")
        print(f"Accuracy: {epoch_acc}\nAP Score: {epoch_ap}\nmAP Score: {epoch_mean_ap}")

        model_training_losses.append(epoch_training_loss)
        model_val_losses.append(epoch_val_loss)
        model_accuracies.append(epoch_acc)
        model_ap_scores.append(epoch_ap)
        model_mean_ap_scores.append(epoch_mean_ap)
        
    training_losses.append(model_training_losses)
    val_losses.append(model_val_losses)
    accuracies.append(model_accuracies)
    ap_scores.append(model_ap_scores)
    mean_ap_scores.append(model_mean_ap_scores)

    models.append(model) # Storing model
    config["learningRate"] *= 5 # New hyperparameter setting for learning rate

# Selecting and saving model with highest validation mAP
best_model = best_training_loss = best_val_loss = best_accuracy = best_ap = best_mean_ap = None
best_idx = 0
temp_mean_ap = np.array(mean_ap_scores[best_idx])

for i in range(len(models)):
    mean_ap = np.array(mean_ap_scores[i])
    if np.max(mean_ap) > np.max(temp_mean_ap):
        temp_mean_ap = mean_ap
        best_idx = i

best_model = models[best_idx]
best_training_loss = training_losses[best_idx]
best_val_loss = val_losses[best_idx]
best_accuracy = accuracies[best_idx]
best_ap = ap_scores[best_idx]
best_mean_ap = mean_ap_scores[best_idx]
torch.save(best_model, os.path.join(os.path.dirname(os.path.abspath(__file__)), "results/best_model.pt"))

# Plotting metrics
plt.plot(range(config["epochs"]), best_accuracy, label="Accuracy")
plt.plot(range(config["epochs"]), best_mean_ap, label="mAP Score")
plt.title(f"Model metrics for lr={config['learningRate']}")
plt.xlabel("Epochs")
plt.ylabel("Metric score")
plt.legend()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results/model_metrics.png"))
plt.clf()

# Plotting loss
plt.plot(range(config["epochs"]), best_training_loss, label="Training loss")
plt.plot(range(config["epochs"]), best_val_loss, label="Validation loss")
plt.title(f"Model loss for lr={config['learningRate']}")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results/model_loss.png"))
plt.clf()


    