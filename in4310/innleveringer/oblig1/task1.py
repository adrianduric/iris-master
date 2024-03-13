from data_mgmt import *
from model import *
import matplotlib.pyplot as plt
from torchvision.models import resnet18


# Setting hyperparameters
config = {
          'seed': 77,
          'use_cuda': True,
          'batch_size': 16,
          'epochs': 10,
          'num_models': 3,
          'learningRate': 1e-3
         }

# Setting seed for testing
torch.manual_seed(config['seed'])

# Creating Datasets and DataLoaders from images
train_dataset, val_dataset, test_dataset, train_dataloader, val_dataloader, test_dataloader = prepare_data(config, seed=config['seed'])

# Storing data from runs
models = []
training_losses = []
val_losses = []
mean_accuracies = []
mean_ap_scores = []
    
# Training and evaluating 3 models
for model_num in range(config["num_models"]):

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
    model_mean_accs = []
    model_mean_ap_scores = []


    # Train model for specified amount of epochs
    for e in range(config["epochs"]):
        print(f"----------- EPOCH {e+1} -----------")
        print("Training model...")
        epoch_training_loss = train_model(train_dataloader, model, loss_fn, optimizer, config)
        print("Training complete.\n")

        # Tracking metrics on validation sets during training
        print("Testing model on validation set...")
        epoch_val_loss, epoch_acc_per_class, epoch_mean_acc, epoch_ap, epoch_mean_ap = test_model(val_dataloader, model, loss_fn, config)
        print("Testing complete.\n")
        print(f"Accuracy per class: {epoch_acc_per_class}\nMean accuracy per class: {epoch_mean_acc}\nAP Score: {epoch_ap}\nmAP Score: {epoch_mean_ap}\n")

        model_training_losses.append(epoch_training_loss)
        model_val_losses.append(epoch_val_loss)
        model_mean_accs.append(epoch_mean_acc)
        model_mean_ap_scores.append(epoch_mean_ap)
        
    training_losses.append(model_training_losses)
    val_losses.append(model_val_losses)
    mean_accuracies.append(model_mean_accs)
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
best_mean_accuracy = mean_accuracies[best_idx]
best_mean_ap = mean_ap_scores[best_idx]
torch.save(best_model, os.path.join(os.path.dirname(os.path.abspath(__file__)), "results/best_model.pt"))

# Plotting metrics
plt.plot(range(config["epochs"]), best_mean_accuracy, label="Mean Accuracy Per Class")
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

# Setting seed again for reproducibility
torch.manual_seed(config['seed'])

# Predicting on test set
print("Testing model on test set...")
logits, labels, indices, _, _, mean_acc, _, mean_ap = test_model(test_dataloader, model, loss_fn, config, get_logits=True)
print("Testing complete.\n")
print(f"Mean accuracy per class: {mean_acc}\nmAP Score: {mean_ap}\n")

#Calculating softmaxes
softmaxes = nn.Softmax(dim=1)(logits)

# Saving softmax scores
torch.save(softmaxes, os.path.join(os.path.dirname(os.path.abspath(__file__)),"results/softmax_scores.pt"))

# Finding 10 best and worst images according to softmax score.
for class_idx in range(3): # Selecting first 3 classes:
    class_indices = []
    class_softmaxes = []

    for i in range(len(labels)):
        if labels[i] == class_idx: # When we find labels of the chosen class:
            class_indices.append(indices[i].int().item()) # Add to collection of these
            class_softmaxes.append(softmaxes[i, class_idx]) #Store corresponding softmax value
        
    # Sorting to find 10 highest and lowest softmax values for given class
    class_indices, class_softmaxes = zip(*sorted(zip(class_indices, class_softmaxes), key=lambda x: x[1], reverse=True))

    best_10_indices = class_indices[:10]
    worst_10_indices = class_indices[-10:]

    # Accessing and saving best and worst images
    figure = plt.figure()
    figure.suptitle(f"Best 10 classifications for class {class_idx}")
    rows = 2
    cols = 5
    for i in range(rows*cols):
        img = test_dataset[best_10_indices[i]][0].numpy()
        figure.add_subplot(rows, cols, i+1)
        plt.imshow(np.einsum("ijk -> jki", img)) 
        plt.axis('off') 
        plt.title(f"Image {i}") 
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),f"results/best10_class{class_idx}"))
    plt.clf()

    figure = plt.figure()
    figure.suptitle(f"Worst 10 classifications for class {class_idx}")
    for i in range(rows*cols):
        img = test_dataset[worst_10_indices[i]][0].numpy()
        figure.add_subplot(rows, cols, i+1)
        plt.imshow(np.einsum("ijk -> jki", img)) 
        plt.axis('off') 
        plt.title(f"Image {i}") 
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),f"results/worst10_class{class_idx}"))
    plt.clf()