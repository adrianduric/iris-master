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

# Loading model and predicting on test set
_, _, _, _, _, test_dataloader = prepare_data(config, seed=config['seed'])
model = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results/best_model.pt"))
loss_fn = nn.CrossEntropyLoss()

print("Test 1...")
old_logits, _, _, _, _, mean_acc, _, mean_ap = test_model(test_dataloader, model, loss_fn, config, get_logits=True)
print("Testing complete.\n")
print(f"Mean accuracy per class: {mean_acc}\nmAP Score: {mean_ap}\n")

#Calculating softmaxes
old_softmaxes = nn.Softmax(dim=1)(old_logits)

# Resetting seed for reproducibility
torch.manual_seed(config['seed'])

# Reloading model and predicting on test set
_, _, _, _, _, test_dataloader = prepare_data(config, seed=config['seed'])
model = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results/best_model.pt"))
loss_fn = nn.CrossEntropyLoss()

print("Test 2...")
new_logits, _, _, _, _, mean_acc, _, mean_ap = test_model(test_dataloader, model, loss_fn, config, get_logits=True)
print("Testing complete.\n")
print(f"Mean accuracy per class: {mean_acc}\nmAP Score: {mean_ap}\n")

#Calculating softmaxes
new_softmaxes = nn.Softmax(dim=1)(new_logits)

# Asserting equality between old and new softmaxes
assert torch.allclose(old_softmaxes, new_softmaxes)
