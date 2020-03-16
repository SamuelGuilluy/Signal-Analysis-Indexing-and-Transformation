import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
import matplotlib.pyplot as plt
from NADE_mine.model import NADE
from python_speech_features import mfcc
from torch.autograd import Variable
import musicnet
import sys, os, signal
import numpy as np
from time import time
from IPython.display import Audio


def train(train_set,train_loader, loss_function, optimizer, model, device):
    # put the train mode
    model.train()
    # initialize the loss
    total_loss = 0.0
    # iterate on the train set
    with train_set:
        for i, (x, y) in enumerate(train_loader):
            # print(x.shape)
            inputs_y = y
            optimizer.zero_grad()

            # print(inputs_y.shape)
            # print(type(inputs_y))

            inputs_y = inputs_y.float().to(device)
            # print(inputs.size())
            # inputs = Variable(inputs.cuda(), requires_grad=False)
            # y = Variable(y.cuda(), requires_grad=False)
            # print(x.size(0))
            y_hat = model(inputs_y)
            # print(x_hat.shape)
            # print(x.shape)
            # loss_functions = nn.CrossEntropyLoss()
            
            # print(y_hat.shape)
            # print(inputs_y.shape)

            loss = loss_function(y_hat, inputs_y)

            #print('y_hat : ', y_hat)

            loss.backward()
            optimizer.step()
            # record
            total_loss += loss.item()
            if i % 10 == 0:
                print(total_loss/(i+1))
    return total_loss


def test(train_set, test_set, test_loader, loss_function, model, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        with train_set, test_set:
            for i, (x, y) in enumerate(test_loader):
                inputs_y = y
                inputs_y = inputs_y.float().to(device)
                # print(inputs_y.shape)
                # print(type(inputs_y))
                # preprocess to binary
                y_hat = model(inputs_y)

                loss = loss_function(y_hat, inputs_y)
                total_loss += loss.item()
            
        print(f"\t[Test Result] loss: {total_loss/len(test_loader.dataset):.4f}")
    return total_loss/len(test_loader.dataset)

    
def non_decreasing(L):
    """for early stopping"""
    return all(x <= y for x, y in zip(L, L[1:]))


data_path = Path(".").absolute().parent / "data"
device = "cuda" if torch.cuda.is_available() else "cpu"
n_step = 3
root = 'musicnet'


def worker_init(args):
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # ignore signals so parent can handle them
    np.random.seed(os.getpid() ^ int(time()))  # approximately random seed for workers


batch_size = 100
kwargs = {'num_workers': 0, 'pin_memory': True, 'worker_init_fn': worker_init}
window = 16384

train_set = musicnet.MusicNet(root=root, train=True, download=True, window=window)
test_set = musicnet.MusicNet(root=root, train=False, window=window)
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, **kwargs)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, **kwargs)

print('data loaded ')
model = NADE(input_dim=128, hidden_dim=50).to(device)
loss_function = nn.BCELoss(reduction="sum")
optimizer = optim.Adam(model.parameters())
scheduler = optim.lr_scheduler.StepLR(optimizer, 15, gamma=0.1)

# start main
train_losses = []
test_losses = []
best_loss = 99999999
wait = 0
for step in range(n_step):
    print(f"Running Step: [{step+1}/{n_step}]")
    train_loss = train(train_set, train_loader, loss_function, optimizer, model, device)
    test_loss = test(train_set, test_set, test_loader, loss_function, model, device)
    scheduler.step()
    # sampling
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    if test_loss <= best_loss:
        best_loss = test_loss
        torch.save(model.state_dict(), "nade-binary.pt")
        print(f"\t[Model Saved]")
        if (step >= 2) and (wait <= 3) and (non_decreasing(test_losses[-3:])):
            wait += 1
        elif wait > 3:
            print(f"[Early Stopped]")
            break
        else:
            continue
