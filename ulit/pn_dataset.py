import os
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat

class CustomDataset(Dataset):
    def __init__(self, data_folder, selected_channels=None):
        self.data_folder = data_folder
        self.classes = sorted(os.listdir(data_folder))
        self.label_to_index = {label: idx for idx, label in enumerate(self.classes)}
        self.selected_channels = selected_channels
        # Load all data paths and corresponding labels
        self.data_paths = []
        self.labels = []
        for idx, label in enumerate(self.classes):
            label_folder = os.path.join(data_folder, label)
            data_files = [f for f in os.listdir(label_folder) if f.endswith('.mat')]
            for data_file in data_files:
                self.data_paths.append(os.path.join(label_folder, data_file))
                self.labels.append(idx)
    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        data_path = self.data_paths[idx]
        label = self.labels[idx]
        # Load the data from .mat file
        data = loadmat(data_path)['window']
        # Select the specified channels
        if self.selected_channels:
            data = data[self.selected_channels]
        # Convert label to one-hot encoding
        return torch.tensor(data).float(), label

