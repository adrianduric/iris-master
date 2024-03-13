from data_mgmt import *
from model import *
import matplotlib.pyplot as plt
from torchvision.models import resnet18


# Setting hyperparameters
config = {
          'seed': 77,
          'use_cuda': True,
          'batch_size': 16,
          'epochs': 1,
          'num_models': 1,
          'learningRate': 1e-3
         }

# Setting seed for testing
torch.manual_seed(config['seed'])

# Loading Datasets, DataLoaders and model
train_dataset, val_dataset, test_dataset, train_dataloader, val_dataloader, test_dataloader = prepare_data(config, seed=config['seed'])
model = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results/best_model.pt"))
if config["use_cuda"]:
    model = model.to("cuda")

# Task 3a)
# Defining hook to perform PCA and calculate projection
chosen_layer = "fc"
projections = []

def hook(module, input, output):
    U, S, V = torch.pca_lowrank(input[0])
    projection = torch.matmul(input[0], V[:, :2])
    projections.append(projection.detach().cpu().numpy())

# Registering hook on last layer (fc)
for nam, mod in model.named_modules():
    if nam == chosen_layer:
        mod.register_forward_hook(hook)

all_labels = torch.Tensor()
if config["use_cuda"]:
    all_labels = all_labels.to("cuda")

# Performing PCA on validation set
for batch_idx, (batch_images, batch_labels, _) in enumerate(val_dataloader):

    if config["use_cuda"]:
        batch_images = batch_images.to("cuda")
        batch_labels = batch_labels.to("cuda")
        
    # Forward pass
    batch_logits = model(batch_images)
    all_labels = torch.cat((all_labels, batch_labels), dim=0)

all_projections = np.vstack(projections)
all_labels = all_labels.cpu().numpy()

# Visualization
plt.figure(figsize=(8, 6))
scatter = plt.scatter(all_projections[:, 0], all_projections[:, 1], c=all_labels, cmap='viridis', alpha=0.6)
plt.colorbar(scatter)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Projection of Feature Space')
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results/pca_best_model.png"))
plt.clf()

# Creating new untrained model and performing same procedure on it
model = resnet18()
model.fc = nn.Linear(in_features=512, out_features=6, bias=True)
if config["use_cuda"]:
    model.to("cuda")

projections = []

for nam, mod in model.named_modules():
    if nam == chosen_layer:
        mod.register_forward_hook(hook)

all_labels = torch.Tensor()
if config["use_cuda"]:
    all_labels = all_labels.to("cuda")

for batch_idx, (batch_images, batch_labels, _) in enumerate(val_dataloader):

    if config["use_cuda"]:
        batch_images = batch_images.to("cuda")
        batch_labels = batch_labels.to("cuda")
        
    batch_logits = model(batch_images)
    all_labels = torch.cat((all_labels, batch_labels), dim=0)

all_projections = np.vstack(projections)
all_labels = all_labels.cpu().numpy()

plt.figure(figsize=(8, 6))
scatter = plt.scatter(all_projections[:, 0], all_projections[:, 1], c=all_labels, cmap='viridis', alpha=0.6)
plt.colorbar(scatter)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Projection of Feature Space')
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results/pca_untrained_model.png"))
plt.clf()
