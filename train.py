from __future__ import print_function
import torch.optim as optim
import argparse
from torchvision import transforms
import torch
from model import ae
from utils import EarlyStopper
import dataset
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

def train(train_list, val_list, test_list, batch_size, learning_rate, total_epochs, use_cuda):
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(dataset.listDataset(train_list,
                                                                   shuffle=True,
                                                                   transform=transforms.Compose(
                                                                       [transforms.Resize((128, 128)),
                                                                        transforms.ToTensor(),]),
                                                                   cell_size=32), batch_size=batch_size, **kwargs)
    val_loader = torch.utils.data.DataLoader(dataset.listDataset(val_list,
                                                                  shuffle=True,
                                                                  transform=transforms.Compose(
                                                                      [transforms.Resize((128, 128)),
                                                                       transforms.ToTensor(),]),
                                                                 cell_size=32), batch_size = batch_size, **kwargs)

    test_loader = torch.utils.data.DataLoader(dataset.listDataset(test_list,
                                                                  shuffle=True,
                                                                  transform=transforms.Compose(
                                                                      [transforms.Resize((128, 128)),
                                                                       transforms.ToTensor(),]),
                                                                 cell_size=32), batch_size = 1, **kwargs)
    writer = SummaryWriter()
    model = ae()
    model.train()
    # if torch.cuda.device_count() > 1:
    # print("Let's use", torch.cuda.device_count(), "GPUs!")
    # model = torch.nn.DataParallel(model)

    criterion = torch.nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.99)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40], gamma=0.1)

    early_stopping = EarlyStopper(patience = 2, min_delta=0.001)
    k = 0
    for epoch in range(total_epochs):
        avg_loss = 0
        for train_idx, train_img in enumerate(train_loader):
            optimizer.zero_grad()
            if use_cuda:
                train_img = train_img.cuda()

            _, train_out = model(train_img)

            train_loss = criterion(train_img, train_out)
            avg_loss += train_loss

            print('epoch:{}, batch: {}, lr: {}, loss: {}'.format(epoch, train_idx, scheduler.get_lr(), train_loss))
            train_loss.backward()
            optimizer.step()
        scheduler.step()
        writer.add_scalar("Loss/train", avg_loss / train_idx, epoch)
        if epoch % 2 == 0:
            print('start validating...')
            with torch.no_grad():
                valid_loss = 0
                for val_idx, val_img in enumerate(val_loader):
                    if use_cuda:
                        val_img = val_img.cuda()
                    _, val_out = model(val_img)
                    valid_loss += criterion(val_img, val_out)
                    writer.add_scalar("Loss/val", valid_loss / val_idx, k)
                is_early_stopping = early_stopping.early_stop(model, valid_loss)
                k += 1

        if is_early_stopping:
            break

    writer.flush()

    model.eval()

    print('testing...')
    for test_idx, test_img in enumerate(test_loader):
        _, test_out = early_stopping.best_model(test_img)
        test_out = test_out.detach().numpy()
        test_out = np.squeeze(test_out)
        test_out = np.transpose(test_out, [1, 2, 0])
        plt.imshow(test_out)
        plt.show()
        break
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='autoencoder-anomaly-segmentation')
    parser.add_argument('--train_list', type=str, default='train_set.txt')
    parser.add_argument('--val_list', type=str, default='val_set.txt')
    parser.add_argument('--test_list', type=str, default='test_set.txt')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--total_epochs', type=int, default = 50)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--use_cuda', type=bool, default=False)

    args = parser.parse_args()
    train_list = args.train_list
    val_list = args.val_list
    test_list = args.test_list
    batch_size = args.batch_size
    use_cuda = args.use_cuda
    total_epochs = args.total_epochs
    learning_rate = args.learning_rate
    train(train_list, val_list, test_list, batch_size, learning_rate, total_epochs, use_cuda)
