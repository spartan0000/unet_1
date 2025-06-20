import os

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


from sklearn.model_selection import train_test_split


#other imports

from build_dataloader import FoggyDataSet, train_test_splt, build_data_loader
from unet_model import Unet

#hyperparameters
n_epochs = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = 1e-4


clear_dir = 'D:/flickr30k_clear'
foggy_dir = 'D:/flickr30k_foggy'


train_files, test_files = train_test_splt(clear_dir, foggy_dir)
train_loader, test_loader = build_data_loader(clear_dir, foggy_dir, train_files, test_files)


def main():
    model = Unet().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0


        
        for i, (foggy_imgs, clear_imgs) in enumerate(train_loader):
            foggy_imgs = foggy_imgs.to(device)
            clear_imgs = clear_imgs.to(device)

            #print(foggy_imgs.shape)

            outputs = model(foggy_imgs)
            loss = criterion(outputs, clear_imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() #keep a running total 
            
            if i % 100 == 0:
                print(f'Batch {i}| Loss: {loss.item():.4f}')

            #print(f'Epoch [{epoch + 1}/{n_epochs}]')

        avg_train_loss=train_loss/len(train_loader)

        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for foggy_imgs, clear_imgs in test_loader:
                foggy_imgs = foggy_imgs.to(device)
                clear_imgs = clear_imgs.to(device)

                outputs = model(foggy_imgs)
                loss = criterion(outputs,clear_imgs)
                val_loss += loss.item()
        avg_val_loss = val_loss/len(test_loader)
        
        print(f'Completed [{epoch + 1}/{n_epochs}] - Training loss: {avg_train_loss:.4f}: Validation loss: {avg_val_loss:.4f} ')

        torch.save(model.state_dict(), 'unet_model_checkpoints')
        
if __name__ == '__main__':     
    main()
