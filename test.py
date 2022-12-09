from __future__ import print_function
import torch.optim as optim
import argparse
from torchvision import transforms
import torch
from torch import nn
from model import ae
from utils import construct_raw_memory_bank, get_new_feature, construct_memory_bank_with_nei
import dataset
import numpy as np
import matplotlib.pyplot as plt
import cv2

raw_bank = construct_raw_memory_bank('train_val_set.txt')
new_bank = construct_memory_bank_with_nei(raw_bank, 3)


test_loader = torch.utils.data.DataLoader(dataset.listDataset('test_set.txt',
                                                                   shuffle=True,
                                                                   test = True,
                                                                   transform=transforms.Compose(
                                                                       [transforms.Resize((128, 128)),
                                                                        transforms.ToTensor(), ]),
                                                                   cell_size=32), batch_size=1)
model = ae()
model.load_state_dict(torch.load('best_model.pt'))
model.eval()
for test_idx, test_img in enumerate(test_loader):
    raw_feature, out = model(test_img)
    new_feature = construct_memory_bank_with_nei(raw_feature, 3)
    new_feature = get_new_feature(new_feature, new_bank, raw_feature, raw_bank)
    out = model.decoder(new_feature)

    out = np.squeeze(out.detach().numpy())
    out = np.transpose(out, [1, 2, 0])

    test_img = np.squeeze(test_img.detach().numpy())
    test_img = np.transpose(test_img, [1, 2, 0])

    diff = out - test_img

    #test_label = np.squeeze(test_label.detach().numpy())
    #test_label = np.transpose(test_label, [1, 2, 0])

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(out, cv2.COLOR_RGB2GRAY))
    plt.title('Decoded image after memory bank')
    plt.subplot(1, 3, 2)
    plt.title('raw image')
    plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY))
    plt.subplot(1, 3, 3)
    plt.title('residual')
    plt.imshow(cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY))
    #plt.subplot(2, 2, 4)
    #plt.imshow(cv2.cvtColor(test_label, cv2.COLOR_RGB2GRAY))
    plt.show()

    break