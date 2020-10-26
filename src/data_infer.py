from torch.utils.data import Dataset
import torchvision.transforms as transforms
import h5py
from param import args
from tok import Tokenizer
from utils import BufferLoader
import copy
from PIL import Image
import json
import random
import os
import numpy as np
import torch

from pathlib import Path
from fastai.vision.all import *   # just need get_image_files, actually

DATA_ROOT = "dataset/"

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def pil_saver(img, path):
    # print(img, path)
    with open(path, 'wb') as f:
        img.save(f)

class InferenceDataset:
    def __init__(self, ds_name='inference', split='test', task='speaker', tok_name='adobe'):
        self.ds_name = ds_name
        self.split = split
        self.source = Path(DATA_ROOT)/ds_name/"source"
        self.crappified = Path(DATA_ROOT)/ds_name/"crappified"

        # Will be ultimately replaced with the one loaded with the saved model,
        # but we need it to create the Speaker decoder instance.
        self.tok = Tokenizer()
        self.tok.load(os.path.join(DATA_ROOT, tok_name, "vocab.txt"))

    def source_for_crap(self, item):
        return self.source/item.relative_to(self.crappified)

class TorchDataset(Dataset):
    def __init__(self, dataset, task='speaker', max_length=80, 
                 img0_transform=None, img1_transform=None):
        self.dataset = dataset
        self.name = dataset.ds_name + "_" + dataset.split
        self.task = task
        self.max_length = max_length
        self.img0_trans, self.img1_trans = img0_transform, img1_transform
        self.train_data = get_image_files(dataset.crappified)

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item):
        img1_path = self.train_data[item]
        img0_path = self.dataset.source_for_crap(img1_path)
        
        # Load Image
        img0 = self.img0_trans(pil_loader(img0_path))   # 3 x 224 x 224
        img1 = self.img1_trans(pil_loader(img1_path))

        return item, img0, img1
