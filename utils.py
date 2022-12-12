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
    """
    :param train_val_list: training sample list
    :param use_cuda: use cuda
    :return: raw_memory_bank, [nsamples, 128, 32, 32]
    """
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

    nsample = 1000
    # 1000 is the train set + val set
    raw_memory_bank = torch.zeros((nsample, 128, 32, 32))
    for train_idx, train_img in enumerate(train_val_loader):
        if train_idx == nsample:
            break
        feature, _ = model(train_img)
        if train_idx % 200 == 0:
            print('constructing memory bank, {}'.format(train_idx % 1000))
        raw_memory_bank[train_idx] = feature
    return raw_memory_bank

def construct_aggregate_memory_bank(memory_bank, w):
    """
    :param memory_bank: raw memory bank, [nsamples, 128, 32, 32]
    :param w: window size
    :return: aggreated memory bank, [nsample, 128 * w * w, 32, 32]
    """
    shape = memory_bank.size()
    agg_bank = torch.zeros((shape[0], shape[1] * w * w, shape[2], shape[3]))
    pads = (w//2, w//2)
    dims = (1, 2)
    for i in range(shape[0]):
        raw_feature = memory_bank[i, :, :, :]
        raw_feature = raw_feature.detach().numpy()
        raw_feature = np.pad(raw_feature, ((0, 0), pads, pads), 'edge')
        agg_feature = np.lib.stride_tricks.sliding_window_view(raw_feature, (w, w), dims)  # apply sliding window on 1 and 2 axis
        agg_bank[i, :, :, :] = torch.Tensor(np.reshape(agg_feature, [shape[1] * w * w, shape[2], shape[3]]))
    print('aggregate bank size: {}'.format(agg_bank.size()))
    return agg_bank


def get_new_feature(agg_feature, agg_bank, raw_feature, raw_bank):
    """
    :param agg_feature: aggreated feature for a test image, [1, 128 * w * w, 32, 32]
    :param agg_bank: aggregated memory bank generated from training image, [nsamples, 128 * w * w, 32, 32]
    :param raw_feature: raw feature from the encoder, [1, 128, 32, 32]
    :param raw_bank: raw memory bank from training image , [nsamples, 128, 32, 32]
    :return: new feature without anomaly, , [128, 32, 32]
    """
    agg_shape = agg_bank.size()
    raw_shape = raw_bank.size()

    # vectorize the memory bank
    raw_bank = raw_bank.view(raw_shape[0] * raw_shape[2] * raw_shape[3], raw_shape[1])
    agg_bank = agg_bank.view(agg_shape[0] * agg_shape[2] * agg_shape[3], agg_shape[1])

    # define cosine similarity
    cos = nn.CosineSimilarity(dim = 1)

    out_feature = torch.zeros((128, 32, 32))
    for i in range(raw_shape[2]):
        print('i: {}'.format(i))
        for j in range(raw_shape[3]):
            arr1 = agg_feature[0, :, i, j]

            # add extra dimension, and repeat
            arr1 = arr1[None, :]
            arr1 = arr1.repeat(agg_shape[0] * agg_shape[2] * agg_shape[3], 1)

            # calculate cosine similarity
            sim_scores = cos(arr1, agg_bank)

            # select top k similar features
            scores, indexes = torch.topk(sim_scores, 10)

            # update the feature
            if torch.mean(scores) < 0.2:
                out_feature[:, i, j] = torch.mean(raw_bank[indexes])
            else:
                out_feature[:, i, j] = raw_feature[0, :, i, j]

    return out_feature


