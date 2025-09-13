###Loading flickr30k dataset from HF


import os
from datasets import load_dataset
from datasets import get_dataset_config_names

import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split

import shutil


#set the cache directory - used external hard drive in this case
HF_CACHE_DIR = 'D:/hf_cache'
dataset_name = "flickr30k"




dataset = load_dataset("flickr30k", cache_dir = "D:/hf_cache/huggingface/datasets")['test']

output_dir = "D:/flickr30k_preprocessed"

os.makedirs(output_dir, exist_ok = True)

#resize all the images to a standard size
#this might distort some of the images because they're not all the same size

resize_transform = transforms.Compose([
    transforms.Resize((256,256)),
])
def resize_images(dataset):
    for i, example in enumerate(dataset):
        img = resize_transform(example['image'])
        filename = os.path.join(output_dir, f'img_{i:05d}.jpg')
        img.save(filename)

        with open(os.path.join(output_dir, f'img_{i:05d}.txt'), 'w', encoding = 'utf-8') as f:
            for caption in example['caption']:
                
                f.write(caption.strip() + "\n")

        if i % 1000 == 0:
            print(f'saved {i} images')

    print('Done resizing images')




#foggy image generator

input_dir = "D:/flickr30k_preprocessed"
output_dir = "D:/flickr30k_foggy"

os.makedirs(output_dir, exist_ok = True)



def add_fog(img_tensor, fog_strength = 0.3):
    fog = torch.empty_like(img_tensor).uniform_(fog_strength, 1)
    foggy_image = img_tensor * (1 - fog_strength) + fog * fog_strength
    return torch.clamp(foggy_image, 0.0, 1.0)

def foggy_image_generator(input_dir, output_dir):
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    for filename in tqdm(os.listdir(input_dir)):
        if not filename.endswith('.jpg'):
            continue
        img_path = os.path.join(input_dir, filename)
        img = Image.open(img_path).convert('RGB')

        img_tensor = to_tensor(img)
        foggy_tensor = add_fog(img_tensor, fog_strength = 0.6)
        foggy_img = to_pil(foggy_tensor)

        name, ext = os.path.splitext(filename)
        foggy_filename = name + '_foggy' + ext
        foggy_img.save(os.path.join(output_dir, foggy_filename))


#gaussian noise generator - add gaussian noise to foggy images

input_dir = "D:/flickr30k_foggy"
output_dir = "D:/flickr30k_foggy_noisy"

os.makedirs(output_dir, exist_ok = True)


def add_noise(img_tensor, mean = 0.0, std = 0.05):
    noise = torch.randn_like(img_tensor) * std + mean
    noisy_image = img_tensor + noise
    return torch.clamp(noisy_image, 0.0, 1.0)

def gaussian_noise_generator(input_dir, output_dir):
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    for filename in tqdm(os.listdir(input_dir)):
        if not filename.endswith('.jpg'):
            continue
        img_path = os.path.join(input_dir, filename)
        img = Image.open(img_path).convert('RGB')
        img_tensor = to_tensor(img)

        noise_tensor = add_noise(img_tensor)
        noisy_image = to_pil(noise_tensor)
        name, ext = os.path.splitext(filename)
        noisy_filename = name + '_noisy' + ext
        noisy_image.save(os.path.join(output_dir, noisy_filename))


#to make the data loading simpler, move the clear images to their own folder clearly labelled as 'clear'

source_dir = 'D:/flickr30k_preprocessed'
dest_dir = 'D:/flickr30k_clear'

os.makedirs(dest_dir, exist_ok = True)


def move_clear_img(source_dir, dest_dir):

    for filename in os.listdir(source_dir):
        if filename.endswith('.jpg'):
            src_path = os.path.join(source_dir, filename)
            dest_path = os.path.join(dest_dir, filename)
            shutil.copy2(src_path, dest_path)


#train_test_split

clear_dir = 'D:/flickr30k_clear'
foggy_dir = 'D:/flickr30k_foggy'

#each clear image has a paired foggy image such that img001.jpg has a corresponding img001_foggy.jpg
def train_test_splt(clear_dir, foggy_dir):
#getting a list of all filenames that have a paired clear and foggy image.  
#returns filenames for training and testing set

    all_filenames = [f for f in os.listdir(clear_dir) if f.endswith('.jpg') and os.path.exists(os.path.join(foggy_dir, f.replace('.jpg', '_foggy.jpg')))]

    train_files, test_files = train_test_split(all_filenames, test_size = 0.2, random_state = 100)
    return train_files, test_files



