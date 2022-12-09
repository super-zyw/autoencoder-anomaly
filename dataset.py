import numpy as np
import matplotlib.pyplot as plt
import os
import PIL
from PIL import Image, ImageDraw
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import random

class listDataset(Dataset):
    def __init__(self, root, shuffle=True, transform = None, test = False, num_workers=4, cell_size=32):
        # read the list of dataset images
        with open(root, 'r') as file:
            self.lines = file.readlines()
        if shuffle:
            random.shuffle(self.lines)
        self.nSamples = len(self.lines)
        self.test = test
        #self.batch_size = batch_size
        self.num_workers = num_workers
        self.cell_size = cell_size
        self.transform = transform
    # Get the number of samples in the dataset
    def __len__(self):
        return self.nSamples
    # Get a sample from the dataset
    def __getitem__(self, index):
        # Ensure the index is smaller than the number of samples in the dataset, otherwise return error
        assert index <= len(self), 'index range error'
        # Get the image path
        imgpath = 'dataset/' + self.lines[index].rstrip()
        img = Image.open(imgpath).convert('RGB')

        if not self.test:
            img = self.transform(img)
            return img
        else:
            w, h = 228,228
            shape = [(40, 40), (w - 10, h - 10)]
            # create line image
            img1 = ImageDraw.Draw(img)
            img1.line(shape, fill="black", width=10)
            img.show()
            img = self.transform(img)
            #labelpath = imgpath.replace('test', 'ground_truth')
            #label = Image.open(labelpath).convert('RGB')
            #label = self.transform(label)
            return img