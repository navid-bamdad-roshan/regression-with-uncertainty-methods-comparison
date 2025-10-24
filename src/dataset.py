import numpy as np
import pandas as pd
import torch
import os
import urllib.request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



def load_concrete_dataset(csv_file='data/concrete_data.csv', train=True, train_ratio=0.8, random_state=42):
    if not os.path.exists(csv_file):
        print(f"File not found at {csv_file}. Downloading...")
        url = 'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/concrete.csv'
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)
        try:
            urllib.request.urlretrieve(url, csv_file)
            print("Download complete.")
        except Exception as e:
            print(f"Failed to download the dataset. Error: {e}")
            raise

    data = pd.read_csv(csv_file)
    features = data.iloc[:, :-1].values
    labels = data.iloc[:, -1].values.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, train_size=train_ratio, random_state=random_state
    )

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    y_train_scaled = y_train / 100  # No specific scaling for labels in this case
    y_test_scaled = y_test / 100  # No specific scaling for labels in this case

    return Dataset(features=X_train_scaled, labels=y_train_scaled), Dataset(features=X_test_scaled, labels=y_test_scaled)
    

    

class Dataset():
    def __init__(self, features, labels):

        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        return x, y
    




class DataLoader():
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset 
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(dataset)
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))
        
    def __len__(self):
        return self.num_batches 
    
    def __iter__(self):
        indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(indices)
        self.indices = indices
        self.current_idx = 0
        return self

    def __next__(self):
        start_idx = self.current_idx
        if start_idx >= self.num_samples:
            raise StopIteration
        end_idx = start_idx + self.batch_size
        if end_idx > self.num_samples:
            end_idx = self.num_samples
        batch_indices = self.indices[start_idx:end_idx]
        x, y = self.dataset.__getitem__(batch_indices)
        self.current_idx = end_idx
        return x, y


