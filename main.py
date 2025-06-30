import os
from symbol import eval_input

os.environ['HF_HUB_CACHE'] = ''
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import torch
torch.cuda.set_device(2)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import stats as st
import torchvision.models as models

from PIL import Image
from torchvision import transforms
from torch.utils.data import  DataLoader
from tqdm import tqdm
import shutil
# from attack import *
from tools import *
from dataload import *
from MSSFT import MSSFT


def run_attack_and_evaluate(attack, model_names,input_dir, output_dir, device,attack_name,batch_size=20, img_size=224,type='normal'):
    print("[1] loading ------------------------------------------------")
    imagenet_dataset = AdvDataset1(root_dir=input_dir, mode='train', img_size=img_size)
    print("[2] loading sets--------------------------------------------")
    train_loader = DataLoader(imagenet_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print("[3] generating Adv--------------------------------------------")
    output0 = f"{output_dir}_{attack_name}"
    if os.path.exists(output0):
        shutil.rmtree(output0)
    os.makedirs(output0)
    for images, labels, filenames in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        delta = attack(images, labels) 
        adv_images = torch.clamp(images + delta, 0, 1) 
        save_images1(adv_images, filenames, output_dir,attack_name)
    
    print("[4] loading-------------------------------------------------")
    output0 = f"{output_dir}_{attack_name}"
    eval_dataset = AdvDataset1(root_dir=output0, mode='eval', img_size=img_size)
    print("[5] loading eval--------------------------------------------")
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    results = {}
    print(f"[6] Attacing {model_names}")
    
    for name in model_names:
        if name == 'HGD':
            model = load_hgd_model(model_names[0],device='cuda')
            correct, total = 0, 0
            for images, labels, _ in eval_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.no_grad():
                    images = transforms.Resize([299,299])(images)
                    outputs = model(images,defense=True)
                    _, predicted = torch.max(outputs[-1].data, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)
            asr = 100 * (1 - correct / total)
            results[model_names[0]] = asr
            print(f"{model_names[0]} ASR: {asr:.1f}%")
            continue
        if type == 'normal':
            model = load_model(name, device).eval().to(device)
        else:
            model = load_defense_model(weight_file=name,device=device)
        correct, total = 0, 0
        for images, labels, filenames in tqdm(eval_loader):
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                  
        asr = 100 * (1 - correct / total)
        results[name] = asr
        print(f"{name} ASR: {asr:.1f}%")
    return results

def main():
    input_dir = ''
    output_dir_mssft = ''
    model_defense = [
        'HGD','NRP','sin','2020Many' , 'vit_b_adv','ResNet101_adv','ResNet50_Chen2024']
    model_mine = [
        'resnet50', 'inception_v3', 'inception_v4', 'resnext101_32x32d','densenet121','vgg19', 'vit_b_32', 'vit_small_r26_s32_224' ,'swin_s', 'twins_svt_base']
    model_list =model_mine
    type = 'normal'
    # type = 'defense'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epsilon = 16 / 255
    alpha = 1.6 / 255
    epochs = 10
    batch_size = 5
    img_size = 224
    for model in model_list[0:1]:
        model_name = model_mine[0]
        print("Running MSSFT attack......")
        attack_mssft = MSSFT(model_name=model_name,adpt=0.9, epsilon=epsilon, alpha=alpha, epoch=epochs,num_scale=20, num_block=3, device=device,)
        results_mssft = run_attack_and_evaluate(attack_mssft, model_list,input_dir, output_dir_mssft, device=device,attack_name=model_name,batch_size=batch_size, img_size=img_size,type=type)

img_min = 0.0
img_max = 1.0
print(f"GPU counts: {torch.cuda.device_count()}")
print(f"GPU ids: {torch.cuda.current_device()}")

if __name__ == '__main__':
    with torch.cuda.device(0):
        main()

       

