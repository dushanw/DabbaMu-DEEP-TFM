
from modules.utils import *
from modules.custom_activations import *
import numpy as np


def loop(device, loader, model, forward_modelA, weights_H, sPSF, exPSF, criterion, opt, type_= 'train', losses = [], epoch= None, m=1, train_model_iter=1, train_H_iter=0):
    losses_temp = []
    opt_model, opt_H = opt
    for i, (x, y) in enumerate(loader):
        x= x.to(device)
        if type_ == 'train':
            model.train()
            opt_model.zero_grad()
            opt_H.zero_grad()
            
            X= x.float()
            # train_model
            for _ in range(train_model_iter):
                Ht= sigmoid_custom(weights_H, m)
                yt = forward_modelA(X, Ht, sPSF, exPSF, device)
                X_hat = model(yt)
                loss = criterion(X_hat, X)
                loss.backward(retain_graph=True)
                opt_model.step()

            #train H
            for _ in range(train_H_iter):
                Ht= sigmoid_custom(weights_H, m)
                yt = forward_modelA(X, Ht, sPSF, exPSF, device)
                X_hat = model(yt)
                loss = criterion(X_hat, X)
                loss.backward(retain_graph=True)
                opt_H.step()

        else:
            model.eval()
            with torch.no_grad():
                X= x.float()
                Ht= sigmoid_custom(weights_H, m)
                yt = forward_modelA(X, Ht, sPSF, exPSF, device)
                X_hat = model(yt)
                loss = criterion(X_hat, X)
        losses_temp.append(loss.item())
    losses.append(np.mean(losses_temp))
    return losses, model, opt, X, X_hat, Ht, yt, weights_H

def train(model, forward_modelA, weights_H, sPSF, exPSF, criterion, opt, train_loader, test_loader, device, T=5, epochs=100, show_results_epoch=1, train_model_iter=1, train_H_iter=0, m_inc_epoc=1, m_inc_proc= None):
    if m_inc_proc==None:
        def m_inc_proc(m, epoch):
            return m

    losses_train= []
    losses_test= []
    print(f'device : {device}')

    m=1
    for epoch in range(1, epochs+1):
        if epoch!=1 and epoch%m_inc_epoc==0:m= m_inc_proc(m, epoch)
        print(f'm : {m}')
        losses_train, model, opt, X, X_hat, Ht, yt, weights_H = loop(device, train_loader, model, forward_modelA, weights_H, sPSF, exPSF, criterion, opt, 'train', losses_train, epoch, m, train_model_iter, train_H_iter)
        losses_test, model, opt, X_val, X_hat_val, Ht_val, yt_val, weights_H = loop(device, test_loader, model, forward_modelA, weights_H, sPSF, exPSF, criterion, opt, 'test', losses_test, epoch, m, train_model_iter, train_H_iter)

        if epoch%show_results_epoch==0:
            show_imgs(X_val, Ht_val, X_hat_val, yt_val, losses_train, losses_test, T, epoch)
            show_results(X_hat_val, X_val, epoch)
            