import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import stats as st
import torchvision.models as models
import timm
import os
from PIL import Image
import pandas as pd
from torchvision import transforms
import timm
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import shutil
import matplotlib.pyplot as plt
import torch_dct as dct
import gc

class AdvDataset1(Dataset):
    def __init__(self, root_dir, mode='train', img_size=224, target_class=None, targeted=False):
        self.img_size = img_size
        self.mode = mode
        self.root_dir = root_dir
        self.image_dir = os.path.join(self.root_dir, 'val_rs') if mode == 'train' else self.root_dir  # 
        self.targeted = targeted
        self.labels = self.load_labels(os.path.join(self.root_dir, 'val_rs.csv'))
        self.filenames = list(self.labels.keys())
        self.target_class = target_class

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        img_path = os.path.join(self.image_dir, filename)
        image = Image.open(img_path).convert('RGB')
        image = image.resize((self.img_size, self.img_size))
        image = np.array(image).astype(np.float32) / 255.0 
        image = torch.from_numpy(image).permute(2, 0, 1)  

        label = self.labels[filename]-1
        if isinstance(label, list): 
            return image, torch.tensor(label), filename
        return image, torch.tensor(label), filename

    def load_labels(self, csv_path):
        if self.mode == 'train':
            dev = pd.read_csv(csv_path)
        else:
            dev = pd.read_csv('./val_rs.csv') 
        f2l = {}
        if self.targeted:
            for _, row in dev.iterrows():
                filename = row['filename']
                original_label = row['label']-1
                if self.target_class is not None:
                    target_label = self.target_class
                else:
                    if 'targeted_label' not in row:
                        raise ValueError("errors")
                    target_label = row['targeted_label']-1
                f2l[filename] = [original_label, target_label]
        else:  
            for index, row_data in dev.iterrows():
                file_name = row_data['filename']  
                label = row_data['label']
                f2l[file_name] = label
        return f2l
