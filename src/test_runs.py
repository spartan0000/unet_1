#some shorter test runs to make sure everything works as expected before training on the full dataset

import os
from dotenv import load_dotenv
from datetime import datetime
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms

import wandb

from PIL import Image
import numpy as np

from timeit import default_timer as timer


from sklearn.model_selection import train_test_split


#other imports

from build_dataloader import FoggyDataSet, FoggyNoisyDataSet, train_test_split_foggy, train_test_split_foggy_noisy, build_foggy_data_loader, build_foggy_noisy_data_loader
from unet_model import Unet, Unet_lite

#hyperparameters and other settings
n_epochs = 2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = 1e-4

experiment_date = datetime.now().strftime('%Y-%m-%d : %H-%M-%S')
config = {
            'learning_rate': lr,
            'epochs': n_epochs,
            'batch_size': 32,
            'optimizer': 'Adam',
            'loss_function': 'MSELoss',
            'model': 'UnetFull',
            'dataset': 'Foggy and Noisy',
            'device': device,
            'subset': 'yes',
            'train_subset_size': 5000,
            'test_subset_size': 2500,

        
        }


#weights and biases setup and login
load_dotenv()
WANDB_API_KEY = os.environ.get('WANDB_API_KEY')
wandb.login(key = WANDB_API_KEY)






#directories containing the data
clear_dir = 'D:/flickr30k_clear'
foggy_dir = 'D:/flickr30k_foggy'
noisy_dir = 'D:/flickr30k_foggy_noisy'

#dataloaders for foggy and foggy/noisy data including arguments for subset sizes
train_files, test_files = train_test_split_foggy(clear_dir, foggy_dir)
train_loader, test_loader, train_subset_loader, test_subset_loader = build_foggy_data_loader(clear_dir, foggy_dir, train_files, test_files, 5000, 2500)

noisy_train_files, noisy_test_files = train_test_split_foggy_noisy(clear_dir, noisy_dir)
noisy_train_loader, noisy_test_loader, noisy_train_subset_loader, noisy_test_subset_loader = build_foggy_noisy_data_loader(clear_dir, noisy_dir, noisy_train_files, noisy_test_files, 5000, 2500)



#Test run
model = Unet(3,3).to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(params = model.parameters(), lr = lr)
use_amp = True
scaler = torch.amp.GradScaler('cuda', enabled = use_amp)


def main():
    wandb.init(
        project = 'Defogging and denoising',
        name = f'exp_{experiment_date}',
        config = config
    )

    for epoch in range(1, n_epochs + 1):
        start_time = timer()
        training_loss = 0.0
        train_psnr = 0.0
            
        model.train()
    

        for i, (images, targets) in enumerate(noisy_train_loader):
            images, targets = images.to(device), targets.to(device)

            with torch.autocast(device_type = 'cuda', enabled = use_amp): #mixed precision really made the training time faster
                outputs = model(images)
                loss = loss_fn(outputs, targets)
                
            batch_psnr = 10 * torch.log10(1/loss)
            batch_psnr = batch_psnr.item()
            optimizer.zero_grad()

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            training_loss += loss.item() * images.size(0)
            train_psnr += batch_psnr * images.size(0)

            #if i % 100 == 0:
            #    print(f'Epoch: {epoch} | Batch: {i} | Training loss: {loss.item():.4f} | PSNR: {batch_psnr:.4f} dB')

        training_loss /= len(noisy_train_loader.dataset)
        train_psnr /= len(noisy_train_loader.dataset)

        test_loss = 0.0
        test_psnr = 0.0
        model.eval()
        with torch.inference_mode():
            for j, (images, targets) in enumerate(noisy_test_loader):
                images, targets = images.to(device), targets.to(device)
                test_outputs = model(images)
                loss = loss_fn(test_outputs, targets)

                batch_psnr = 10 * torch.log10(1/loss)
                batch_psnr = batch_psnr.item()

                
                test_loss += loss.item() * images.size(0)
                test_psnr += batch_psnr * images.size(0)

                #if j % 100 == 0:   

                #    print(f'Epoch: {epoch} | Test batch: {j} | Test loss: {loss.item():.4f} | PSNR: {batch_psnr:.4f} dB')
            test_loss /= len(noisy_test_loader.dataset)
            test_psnr /= len(noisy_test_loader.dataset)
        
        print(f' {epoch} / {n_epochs} - Training loss: {training_loss:.4f}| PSNR: {test_psnr:.4f} dB | Test loss: {test_loss:.4f} | PSNR: {test_psnr:.4f} dB')
        
        
        wandb.log({
            'training_loss': training_loss,
            'train_psnr': train_psnr,
            'test_loss': test_loss,
            'test_psnr': test_psnr,
            'epoch': epoch,

        })

        end_time = timer()
        print(f'Total time: {end_time - start_time:.4f} seconds on {device}')

    PATH = 'D:/foggy_noisy_unet.pth'
    torch.save(model.state_dict(), PATH)
if __name__ == '__main__':
    main()