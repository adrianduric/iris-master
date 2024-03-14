from data_mgmt import *
from model import *
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from torchvision.models import resnet18
from sklearn.decomposition import PCA


# Setting hyperparameters
config = {
          'seed': 77,
          'use_cuda': True,
          'batch_size': 16,
          'epochs': 1,
          'num_models': 1,
          'learningRate': 1e-3
         }

# Storing class dict for convenience
classes = {
    0: "buildings",
    1: "forest",
    2: "glacier",
    3: "mountain",
    4: "sea",
    5: "street"
}

# Setting seed for testing
torch.manual_seed(config['seed'])

# Loading Datasets, DataLoaders and model
train_dataset, val_dataset, test_dataset, train_dataloader, val_dataloader, test_dataloader = prepare_data(config, seed=config['seed'])
model = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results/best_model.pt"))
if config["use_cuda"]:
    model = model.to("cuda")
model.eval()

# Task 3a)
# Defining hook to store feature maps for PCA
chosen_layer = "fc"
feature_maps = []

def hook(module, input, output):
    feature_maps.append(input[0].detach().cpu().numpy())

# Registering hook on last layer (fc)
for nam, mod in model.named_modules():
    if nam == chosen_layer:
        mod.register_forward_hook(hook)

all_labels = torch.Tensor()
if config["use_cuda"]:
    all_labels = all_labels.to("cuda")

# Iterating through validation set to collect feature maps
for batch_idx, (batch_images, batch_labels, _) in enumerate(val_dataloader):

    if config["use_cuda"]:
        batch_images = batch_images.to("cuda")
        batch_labels = batch_labels.to("cuda")
        
    # Forward pass
    batch_logits = model(batch_images)
    all_labels = torch.cat((all_labels, batch_labels), dim=0)

# Performing PCA
feature_maps = np.vstack(feature_maps)
pca = PCA(n_components=2)
projections = pca.fit_transform(feature_maps)

all_labels = all_labels.cpu().numpy()

# Visualization
plt.figure(figsize=(8, 6))
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Projection of Feature Space')

colors = plt.cm.viridis(np.linspace(0, 1, 6))
legend_elements = [Line2D([0], [0], marker='o', color='w', label=f'{classes[i]}', markerfacecolor=clr, markersize=10) for i, clr in enumerate(colors)]

for class_num in range(6):  # For each class:
    projections_of_class = projections[all_labels == class_num] 
    plt.scatter(projections_of_class[:, 0], projections_of_class[:, 1], c=[colors[class_num]], alpha=0.6)

plt.legend(handles=legend_elements, loc="best")
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results/pca_best_model.png"))
plt.clf()

# Creating new untrained model and performing same procedure on it
model = resnet18()
model.fc = nn.Linear(in_features=512, out_features=6, bias=True)
if config["use_cuda"]:
    model.to("cuda")
model.eval()

feature_maps = []

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

feature_maps = np.vstack(feature_maps)
pca = PCA(n_components=2)
projections = pca.fit_transform(feature_maps)

all_labels = all_labels.cpu().numpy()

plt.figure(figsize=(8, 6))
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Projection of Feature Space')

colors = plt.cm.viridis(np.linspace(0, 1, 6))
legend_elements = [Line2D([0], [0], marker='o', color='w', label=f'{classes[i]}', markerfacecolor=clr, markersize=10) for i, clr in enumerate(colors)]

for class_num in range(6):  # For each class:
    projections_of_class = projections[all_labels == class_num] 
    plt.scatter(projections_of_class[:, 0], projections_of_class[:, 1], c=[colors[class_num]], alpha=0.6)

plt.legend(handles=legend_elements, loc="best")
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results/pca_untrained_model.png"))
plt.clf()
