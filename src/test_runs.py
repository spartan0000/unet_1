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
train_loader, test_loader, train_subset_loader, test_subset_loader = build_foggy_data_loader(clear_dir, foggy_dir, train_files, test_files, 2000, 1000)

noisy_train_files, noisy_test_files = train_test_split_foggy_noisy(clear_dir, noisy_dir)
noisy_train_loader, noisy_test_loader, noisy_train_subset_loader, noisy_test_subset_loader = build_foggy_noisy_data_loader(clear_dir, noisy_dir, noisy_train_files, noisy_test_files, 2000, 1000)

#subset test run
model_lite = Unet_lite(3,3).to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(params = model_lite.parameters(), lr = lr)

def main():

    for epoch in range(n_epochs):
        start_time = timer()
        training_loss = 0.0

        train_psnr = 0.0
              
        model_lite.train()


        

        for i, (images, targets) in enumerate(noisy_train_subset_loader):
            images, targets = images.to(device), targets.to(device)

            outputs = model_lite(images)
            loss = loss_fn(outputs, targets)
            
            batch_psnr = 10 * torch.log10(1/loss)
            batch_psnr = batch_psnr.item()
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            training_loss += loss.item() * images.size(0)
            train_psnr += batch_psnr * images.size(0)

            if i % 2 == 0:
                print(f'Batch: {i+1} | Training loss: {loss.item():.4f} | PSNR: {batch_psnr:.4f} dB')

        #training_loss /= len(noisy_train_subset_loader.dataset)
        #train_psnr /= len(noisy_train_subset_loader.dataset)

        test_loss = 0.0
        test_psnr = 0.0
        model_lite.eval()
        with torch.inference_mode():
            for j, (images, targets) in enumerate(noisy_test_subset_loader):
                images, targets = images.to(device), targets.to(device)
                test_outputs = model_lite(images)
                loss = loss_fn(test_outputs, targets)

                batch_psnr = 10 * torch.log10(1/loss)
                batch_psnr = batch_psnr.item()

                
                test_loss += loss.item() * images.size(0)
                test_psnr == batch_psnr * images.size(0)

                if j % 2 == 0: #just get output for even numbered batches

                    print(f'Test batch: {j+1} | Test loss: {loss.item():.4f} | PSNR: {batch_psnr:.4f} dB')
            test_loss /= len(noisy_test_subset_loader.dataset)
            test_psnr /= len(noisy_test_subset_loader.dataset)
        
        #print(f' {epoch + 1} / {n_epochs} - Training loss: {training_loss:.4f}| PSNR: {test_psnr:.4f} dB | Test loss: {test_loss:.4f} | PSNR: {test_psnr:.4f} dB')
        
        
        end_time = timer()
        print(f'Total time: {end_time - start_time:.4f} seconds on {device}')


if __name__ == '__main__':
    main()