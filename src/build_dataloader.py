#buliding the dataloader
import os
from datasets import load_dataset
from datasets import get_dataset_config_names

import matplotlib.pyplot as plt

import torch
from torch.utils.data import Subset, Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split

class FoggyDataSet(Dataset):
    def __init__(self, foggy_dir, clear_dir, filenames, transform = None):
        self.foggy_dir = foggy_dir
        self.clear_dir = clear_dir
        self.transform = transform
        self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        

        foggy_path = os.path.join(self.foggy_dir, filename.replace('.jpg', '_foggy.jpg'))
        clear_path = os.path.join(self.clear_dir, filename)

        foggy_image = Image.open(foggy_path).convert('RGB')
        clear_image = Image.open(clear_path).convert('RGB')

        if self.transform:
            foggy_image = self.transform(foggy_image)
            clear_image = self.transform(clear_image)

        return foggy_image, clear_image
    
class FoggyNoisyDataSet(Dataset):
    def __init__(self, foggy_noisy_dir, clear_dir, filenames, transform = None):
        super().__init__()
        self.foggy_noisy_dir = foggy_noisy_dir
        self.clear_dir = clear_dir
        self.filenames = filenames
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        foggy_noisy_path = os.path.join(self.foggy_noisy_dir, filename.replace('.jpg', '_foggy_noisy.jpg'))
        clear_path = os.path.join(self.clear_dir, filename)

        foggy_noisy_image = Image.open(foggy_noisy_path).convert('RGB')
        clear_image = Image.open(clear_path).convert('RGB')

        if self.transform:
            foggy_noisy_image = self.transform(foggy_noisy_image)
            clear_image = self.transform(clear_image)
        return foggy_noisy_image, clear_image 
#train_test_split


#each clear image has a paired foggy image such that img001.jpg has a corresponding img001_foggy.jpg
def train_test_split_foggy(clear_dir, foggy_dir):
#getting a list of all filenames that have a paired clear and foggy image.  
#returns filenames for training and testing set

    all_filenames = [f for f in os.listdir(clear_dir) if f.endswith('.jpg') and os.path.exists(os.path.join(foggy_dir, f.replace('.jpg', '_foggy.jpg')))]

    train_files, test_files = train_test_split(all_filenames, test_size = 0.2, random_state = 100)
    return train_files, test_files

def train_test_split_foggy_noisy(clear_dir, foggy_noisy_dir):
    all_filenames = [f for f in os.listdir(clear_dir) if f.endswith('.jpg') and os.path.exists(os.path.join(clear_dir, f.replace('.jpg', '_foggy_noisy.jpg')))]
    train_files, test_files = train_test_split(all_filenames, test_size = 0.2, randome_state = 100)
    return train_files, test_files


def build_foggy_data_loader(clear_dir, foggy_dir, train_files, test_files, train_subset, test_subset):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_data = FoggyDataSet(foggy_dir, clear_dir, train_files, transform)
    test_data = FoggyDataSet(foggy_dir, clear_dir, test_files, transform)

    train_subset = Subset(train_data, range(train_subset))
    test_subset = Subset(test_data, range(test_subset))

    train_subset_loader = DataLoader(train_subset, batch_size = 32, shuffle = True)
    test_subset_loader = DataLoader(test_subset, batch_size = 32)

    train_loader = DataLoader(train_data, batch_size = 32, shuffle = True)
    test_loader = DataLoader(test_data, batch_size = 32)

    return train_loader, test_loader, train_subset_loader, test_subset_loader


def build_foggy_noisy_data_loader(clear_dir, foggy_noisy_dir, train_files, test_files, train_subset, test_subset):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_data = FoggyNoisyDataSet(foggy_noisy_dir, clear_dir, train_files, transform)
    test_data = FoggyNoisyDataSet(foggy_noisy_dir, clear_dir, test_files, transform)

    train_subset = Subset(train_data, range(train_subset))
    test_subset = Subset(test_data, range(test_subset))

    train_subset_loader, = DataLoader(train_subset, batch_size = 32, shuffle = True)
    test_subset_loader = DataLoader(test_subset, batch_size = 32)

    train_loader = DataLoader(train_data, batch_size = 32, shuffle = True)
    test_loader = DataLoader(test_data, batch_size = 32)

    return train_loader, test_loader, train_subset_loader, test_subset_loader