#some shorter test runs to make sure everything works as expected before training on the full dataset

import os

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

from timeit import default_timer as timer


from sklearn.model_selection import train_test_split


#other imports

from build_dataloader import FoggyDataSet, FoggyNoisyDataSet, train_test_split_foggy, train_test_split_foggy_noisy, build_foggy_data_loader, build_foggy_noisy_data_loader
from unet_model import Unet, Unet_lite

#hyperparameters
n_epochs = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = 1e-4


clear_dir = 'D:/flickr30k_clear'
foggy_dir = 'D:/flickr30k_foggy'
noisy_dir = 'D:/flickr30k_foggy_noisy'


train_files, test_files = train_test_split_foggy(clear_dir, foggy_dir)
train_loader, test_loader, train_subset_loader, test_subset_loader = build_foggy_data_loader(clear_dir, foggy_dir, train_files, test_files, 500, 100)

noisy_train_files, noisy_test_files = train_test_split_foggy_noisy(clear_dir, noisy_dir)
noisy_train_loader, noisy_test_loader, noisy_train_subset_loader, noisy_test_subset_loader = build_foggy_noisy_data_loader(clear_dir, noisy_dir, noisy_train_files, noisy_test_files, 500, 100)

#subset test run
model_lite = Unet_lite(3,3).to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(params = model_lite.parameters(), lr = lr)

def main():

    for epoch in range(n_epochs):
        start_time = timer()
        training_loss = 0.0

        
        model_lite.train()


        

        for images, targets in noisy_train_subset_loader:
            images, targets = images.to(device), targets.to(device)

            outputs = model_lite(images)
            loss = loss_fn(outputs, targets)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        training_loss += loss.item() * images.size(0)
        
        training_loss /= len(noisy_train_subset_loader.dataset)
        test_loss = 0.0
        model_lite.eval()
        with torch.inference_mode():
            for images, targets in noisy_test_subset_loader:
                images, targets = images.to(device), targets.to(device)
                test_outputs = model_lite(images)
                loss = loss_fn(test_outputs, targets)
                test_loss += loss.item() * images.size(0)
            test_loss /= len(noisy_test_subset_loader.dataset)
        end_time = timer()
        print(f' {epoch + 1} / {n_epochs} - Training loss: {training_loss:.4f} | Test loss: {test_loss:.4f}')
        print(f'Total time: {end_time - start_time:.4f} seconds on {device}')


if __name__ == '__main__':
    main()