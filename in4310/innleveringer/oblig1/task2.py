from data_mgmt import *
from model import *


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

# Task 2a)
# Defining hook to extract output of module (feature map)
chosen_layers = ["layer1", "layer2", "layer3", "layer4", "avgpool"]
npv_percentages = []

def hook(module, input, output):
    pos_vals = torch.where(output > 0, output, 0)
    count = torch.count_nonzero(pos_vals).item()
    npv_percentages.append(1 - (count/output.numel()))
        
for nam, mod in model.named_modules():
    if nam in chosen_layers:
        mod.register_forward_hook(hook)

# Task 2b)
# Training model again to activate hooks   
# Iterating through 200 images
num_batches = np.ceil(200 / config["batch_size"])
for batch_idx, (batch_images, batch_labels, _) in enumerate(train_dataloader):
    if config["use_cuda"]:
        batch_images = batch_images.to("cuda")
        batch_labels = batch_labels.to("cuda")
        
    # Forward pass
    batch_logits = model(batch_images)

    if batch_idx >= num_batches - 1:
        break

# Reporting average percentage
npv_avg = np.mean(np.array(npv_percentages))
print(f"Average percentage of non-positive values: {round(npv_avg * 100, 2)}%")
