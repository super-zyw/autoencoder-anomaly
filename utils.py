import numpy as np
import torch
from torch import nn
import dataset
from torchvision import transforms
from model import ae

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.best_model = None

    def early_stop(self, model, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            self.best_model = model
            torch.save(self.best_model.state_dict(), 'best_model.pt')
            print('model saved....')
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def construct_raw_memory_bank(train_val_list, use_cuda = False):
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    train_val_loader = torch.utils.data.DataLoader(dataset.listDataset(train_val_list,
                                                                   shuffle=True,
                                                                   transform=transforms.Compose(
                                                                       [transforms.Resize((128, 128)),
                                                                        transforms.ToTensor(),]),
                                                                   cell_size=32), batch_size=1, **kwargs)

    model = ae()
    model.load_state_dict(torch.load('best_model.pt'))
    model.eval()

    # 1000 is the train set + val set
    raw_memory_bank = torch.zeros((1000, 128, 32, 32))
    for train_idx, train_img in enumerate(train_val_loader):
        feature, _ = model(train_img)
        if train_idx % 200 == 0:
            print('constructing memory bank, {}'.format(train_idx % 1000))
        raw_memory_bank[train_idx] = feature
    return raw_memory_bank

def construct_memory_bank_with_nei(memory_bank, w):
    shape = memory_bank.size()
    new_bank = torch.zeros((shape[0], shape[1] * w * w, shape[2], shape[3]))

    for i in range(shape[2] - w + 1):
        for j in range(shape[3] - w + 1):
            for k in range(shape[0]):
                bank = memory_bank[k, :, i:i+w, j:j+w]
                new_bank[k, :, i, j] = torch.reshape(bank, (-1,))

    for i in range(shape[2] - w + 1, shape[2]):
        for j in range(shape[3] - w + 1, shape[3]):
            for k in range(shape[0]):
                bank = memory_bank[k, :, i-w:i, j-w:j]
                new_bank[k, :, i, j] = torch.reshape(bank, (-1,))
    return new_bank


def get_new_feature(nei_feature, nei_memory_bank, raw_feature, raw_bank):
    # extract_feature : [1, 128, 32, 32]
    # memory_bank: [1000, 128, 32, 32]

    shape = nei_memory_bank.size()
    cos = nn.CosineSimilarity()

    avg_feature = torch.zeros((128, 32, 32))
    for i in range(shape[2]):
        for j in range(shape[3]):
            sim_scores = [[0, 0]] * shape[0]
            for k in range(shape[0]):
                sim_score = cos(torch.unsqueeze(nei_feature[0, :, i, j], dim = 1), torch.unsqueeze(nei_memory_bank[k, :, i, j], 1))
                sim_scores[k] = [sim_score.detach().numpy()[0], k]
            sim_scores = sorted(sim_scores, key=lambda x: x[0], reverse = True)[:10]
            scores = [score for index, score in sim_scores]
            indexes = [index for index, score in sim_scores]
            if np.mean(scores) < 0.8:
                avg_feature[:, i, j] = torch.mean(raw_bank[indexes], dim = 0)[:, i, j]
            else:
                avg_feature[:, i, j] = raw_feature[0, :, i, j]
    return avg_feature