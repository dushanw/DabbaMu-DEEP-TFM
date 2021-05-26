import torch 
from torch import nn
import torchvision
from modules.utils import *
import numpy as np

def loop(n_gpus, loader, model, criterion, opt, type_= 'train', losses = [], epoch= None):
    losses_temp = []
    model = nn.DataParallel(model, device_ids = list(range(n_gpus))).to('cuda:0')
    for i, (x, y) in enumerate(loader):
        x= x.to('cuda:0')
        if type_ == 'train':
            model.train()
            opt.zero_grad()
            x_hat = model(x)
            loss = criterion(x_hat, x)
            loss.backward(retain_graph=True)
            opt.step()
        else:
            model.eval()
            with torch.no_grad():
                x_hat = model(x)
                loss = criterion(x_hat, x)
        losses_temp.append(loss.item())
    losses.append(np.mean(losses_temp))
    return losses, model, opt, x, x_hat

def train(model, criterion, opt, train_loader, test_loader, n_gpus, latent_dim=10, epochs=100, show_results_epoch=1, train_encoder= True):
    losses_train= []
    losses_test= []
    print(f'n_gpus : {n_gpus}')
    model= freeze_encoder(model, freeze= not train_encoder)

    for epoch in range(1, epochs+1):
        losses_train, model, opt, x, x_hat = loop(n_gpus, train_loader, model, criterion, opt, 'train', losses_train, epoch)
        losses_test, model, opt, x_val, x_hat_val = loop(n_gpus, test_loader, model, criterion, opt, 'test', losses_test, epoch)

        if epoch%show_results_epoch==0:
            show_results(x_hat_val, x_val, epoch, losses_train, losses_test)