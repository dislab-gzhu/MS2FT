import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
import timm
import os
from PIL import Image
from torchvision import transforms
import timm
from dataload import *
from collections import OrderedDict
from inres import *

def prepare_comparison_data(*results_dicts, method_names):

    if len(results_dicts) != len(method_names):
        raise ValueError("errors: results_dicts and method_names must have the same length")

    name_mapping = {
        'resnet18': 'ResNet-18',
        'resnet101': 'ResNet-101',
        'inception_v3': 'Inception-v3',
        'inception_v4': 'Inception-v4',
        'resnet50': 'ResNet-50',
        'densenet121': 'DenseNet-121',
        'mobilenet_v2': 'MobileNet',
        'vit_b_16': 'ViT-B',
        'vit_b_32': 'ViT-B-32',
        'vit_l_16': 'VIT-L16',
        'vit_l_32': 'VIT-L',
        'swin_t': 'Swin-T',
        'mobilevit_s': 'MobileViT',
        'vgg16': 'VGG-16',
        'vit_base_patch32_224': 'VIT_B_32',
        'vit_small_r26_s32_224': 'ViT-Res26',
        'vit_base_r50_s16_224': 'VIT_B_R50S16',
        'vit_large_patch16_224': 'VIT_L_16_timm',
        'vgg19': 'VGG-19',
        'vgg11': 'VGG11',
        'vgg16': 'VGG16',
        'twins_pcpvt_base': 'Twins-B',
        'twins_pcpvt_small':'Twins-P-S',
        'twins_svt_base':'Twins-S-B',
        'twins_svt_small':'Twins-S-S',
        'swin_s': 'Swin-S',
        'swin_b': 'Swin-B',
        'vit_base_r26_s32_224':'ViT_B_R26s32', 
        'vit_base_r50_s16_384':'ViT_B_R50s16',
        'vit_base_resnet26d_224':'ViT_B_Re26d',
        'vit_base_resnet50d_224':'ViT_B_Re50d',
        'densenet169':'Desenet169',
        'resnet152':'Resnet-152',
        'resnext50_32x4d':'ResNext-50',
        'resnext101_32x16d':'D',
          'resnext101_32x32d':'ResNext101-32D', 
    }

    models = list(results_dicts[0].keys())

    comparison_data = {}
    for model in models:
        mapped_name = name_mapping.get(model, model) 
        comparison_data[mapped_name] = {
            method: results[model] for method, results in zip(method_names, results_dicts)
        }

    return comparison_data

def clamp(x, x_min, x_max):
    return torch.min(torch.max(x, x_min), x_max)

def save_images1(adversaries, filenames, output_dir,attack_name):
    output = f"{output_dir}_{attack_name}"
    os.makedirs(output, exist_ok=True)
    adversaries = (adversaries.detach().permute((0, 2, 3, 1)).cpu().numpy() * 255).astype(np.uint8)
    for i, filename in enumerate(filenames):
        Image.fromarray(adversaries[i]).save(os.path.join(output, filename))

def wrap_model(model):
    model_name = model.__class__.__name__
    Resize = 224
    if hasattr(model, 'default_cfg'):
        """timm.models"""
        mean = model.default_cfg['mean']
        std = model.default_cfg['std']
    else:
        """torchvision.models"""
        if 'Inc' in model_name:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
            Resize = 299
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            Resize = 224


    PreprocessModel = PreprocessingModel(Resize, mean, std)
    return torch.nn.Sequential(PreprocessModel, model) 

def load_model(model_name,device):
    def load_single_model(model_name):
        if model_name in models.__dict__.keys():
            print(f'=> Loading model {model_name} from torchvision.models')
            model = models.__dict__[model_name](weights="DEFAULT")
        elif model_name in timm.list_models():
            print(f'=> Loading model {model_name} from timm.models')
            model = timm.create_model(model_name, pretrained=True)
        else:
            raise ValueError(f'Model {model_name} not supported')
        return wrap_model(model.eval().to(device))

    if isinstance(model_name, list):  # 检查输入是否为列表，列表即为做集成攻击
        return EnsembleModel([load_single_model(name) for name in model_name])
    else:
        return load_single_model(model_name)


def wrap_model_adv(model, model_name):
    adv_models = ['ResNet50_adv', 'ResNet101_adv','vit_b_adv']
    defense =['HGD']
    if model_name in adv_models:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        preprocess = PreprocessingModel(resize=224, mean=mean, std=std)
        return nn.Sequential(preprocess, model) 
    else:
        return model

def load_defense_model(weight_folder='./defense', weight_file=None, device='cuda'):

    model_mapping = {
        'augmix.pt': lambda: models.resnet50(pretrained=False),
        'cutmix.pth.tar':lambda: models.resnet50(pretrained=False),
        'noisymix.pt': lambda: models.resnet50(pretrained=False),
        'noisymix_new.pt': lambda: models.resnet50(pretrained=False),
        'sin_in.pt': lambda: models.resnet50(pretrained=False),
        'sin.pt': lambda: models.resnet50(pretrained=False),
        'ResNet50_adv.pth': lambda: models.resnet50(pretrained=False),
        'ResNet101_adv.pth': lambda: models.resnet101(pretrained=False),
        'vit_b_adv.pt': lambda: timm.create_model('vit_base_patch16_224', pretrained=False),
        'ResNet50_Chen2024.pt': lambda: models.resnet50(width_per_group=64 * 2,pretrained=False),
        'swin_b_advRM2024.pt':lambda :timm.create_model('swin_base_patch4_window7_224', pretrained=False),
        '2020Many.pt':lambda: models.resnet50(pretrained=False),
        'Salman_R50.pt': lambda: models.resnet50(pretrained=False),
        'ARES_ConvNext_Base_AT.pth':lambda: timm.create_model('convnext_base', pretrained=False),
        'Chen2024Data_rn-50.pt':lambda:models.resnet50(width_per_group=64 * 2,pretrained=False),
        'sin_in_in.pt':lambda: models.resnet50(pretrained=False),
        'ARES_Swin_base_patch4_window7_224_AT.pth':lambda: timm.create_model('swin_base_patch4_window7_224', pretrained=False),
        'Mo2022When_ViT-B.pt':lambda:timm.create_model('vit_base_patch16_224', pretrained=False,),
        'Singh2023Revisiting_ViT-B-ConvStem.pt': lambda: timm.create_model('vit_b_cvst', pretrained=False),
        'Salman2020Do_R18.pt':lambda: models.resnet18(pretrained=False),
        
    }
    candidate_suffixes = ['.pt', '.pth', '.pth.tar','ckpt']
    candidate_keys = [f"{weight_file}{suffix}" for suffix in candidate_suffixes]
    matched_key = None
    for key in candidate_keys:
        if key in model_mapping:
            matched_key = key
            break
    model = model_mapping[matched_key]()
    weight_path = os.path.join(weight_folder, matched_key)
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"权重文件不存在：{weight_path}")
    checkpoint = torch.load(weight_path, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model = wrap_model_adv(model, weight_file)
    model.eval()
    model.to(device)
    print(f"loading success：{matched_key}")
    return model

class NormalizeLayer(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeLayer, self).__init__()
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def forward(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

class PreprocessingModel(nn.Module):
    def __init__(self, resize, mean, std):
        super(PreprocessingModel, self).__init__()
        self.resize = transforms.Resize(resize,antialias=True)
        self.normalize = transforms.Normalize(mean, std)

    def forward(self, x):
        return self.normalize(self.resize(x))


class EnsembleModel(torch.nn.Module):
    def __init__(self, models, mode='mean'):
        super(EnsembleModel, self).__init__()
        self.device = next(models[0].parameters()).device
        for model in models:
            model.to(self.device)
        self.models = models
        self.softmax = torch.nn.Softmax(dim=1)
        self.type_name = 'ensemble'
        self.num_models = len(models)
        self.mode = mode

    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        outputs = torch.stack(outputs, dim=0)
        if self.mode == 'mean':
            outputs = torch.mean(outputs, dim=0)
            return outputs
        elif self.mode == 'ind':
            return outputs
        else:
            raise NotImplementedError

def load_hgd_model(model_name,device='cuda'):
    config, inresmodel = get_model()
    model = inresmodel.net
    checkpoint = torch.load('./.ckpt')
    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('net.', '') 
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    model.to(device)
    print(f"loading success {model_name}")
    return model
