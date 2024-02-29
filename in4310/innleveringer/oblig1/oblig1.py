import os
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

# Task 1a)
# Setting X and y as features and targets
X = []
y = []

images_dir = "mandatory1_data"

for label_dir, _, filenames in os.walk(images_dir):
    label = label_dir

    for filename in filenames:
        file_path = os.path.join(label_dir, filename)
        img = Image.open(file_path)
        
        X.append(np.array(img))
        y.append(label)

# Performing split of dataset
# First splitting test set and the rest
TEST_PORTION = 3000/len(X) # splits approx. 3000 of features to test set
sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_PORTION)

for temp_index, test_index in sss.split(X, y):
    X_temp, y_temp = [X[i] for i in temp_index], [y[i] for i in temp_index]
    X_test, y_test = [X[i] for i in test_index], [y[i] for i in test_index]

# Then splitting training and validation sets
VAL_PORTION = 2000/len(X_temp) # approx. 2000 features in validation set
sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_PORTION)

for train_index, val_index in sss.split(X_temp, y_temp):
    X_train, y_train = [X_temp[i] for i in train_index], [y_temp[i] for i in train_index]
    X_val, y_val = [X_temp[i] for i in val_index], [y_temp[i] for i in val_index]

# Asserting that sets are disjointed
assert set(train_index).isdisjoint(set(val_index))
assert set(train_index).isdisjoint(set(test_index))
assert set(val_index).isdisjoint(set(test_index))




















