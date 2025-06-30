import torch
import torch.nn as nn
import random
import numpy as np
import torchvision.transforms as T
from tools import *
class MSSFT:
    def __init__(self, model_name,adpt, num_scale,epsilon=16 / 255, alpha=1.6 / 255, epoch=10, decay=1.0, num_block=3,
                 targeted=False, random_start=False, norm='linfty', loss='crossentropy', device=None):
        if norm not in ['l2', 'linfty']:
            raise ValueError(f"Unsupported norm {norm}")
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(model_name, self.device)
        self.epsilon = epsilon
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay
        self.num_scale = num_scale
        self.num_block = num_block
        self.targeted = targeted
        self.random_start = random_start
        self.norm = norm
        self.adpt = adpt
        self.loss = nn.CrossEntropyLoss() if loss == 'crossentropy' else None


    def forward(self, data, label):
        # Note that, core functions will be updated once the paper has been accepted for publication.

    def __call__(self, *input, **kwargs):
        self.model.eval()
        return self.forward(*input, **kwargs)