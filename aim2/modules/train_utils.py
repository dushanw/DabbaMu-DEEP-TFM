
from modules.utils import *
from modules.custom_activations import *
import numpy as np
import time


from modules.models.classifiers import classification_accuracy as accuracy

def evaluate(device, loader, model, forward_modelA, modelH, classifier, sPSF, exPSF, m, noise=None, rescale=[-1, 1], noise_K=1):
    if rescale!=None:rescale_mean , rescale_std = rescale
    else:rescale_mean , rescale_std= 0, 1
    
    acc_on_real=[]
    acc_on_fake=[]
    for i, (x, y) in enumerate(loader):
        x= x.to(device)
        y= y.to(device)
        model.eval()
        modelH.eval()
        with torch.no_grad():
            X= x.float()
            Ht= modelH(m)
            yt = forward_modelA(X, Ht, sPSF, exPSF, device, noise, noise_K)
            X_hat = model(yt)
            
            acc_on_real.append(accuracy(y, classifier(X*rescale_std+rescale_mean)))
            acc_on_fake.append(accuracy(y, classifier(X_hat*rescale_std+rescale_mean)))
            
    return np.mean(acc_on_real), np.mean(acc_on_fake)


def loop(device, loader, model, forward_modelA, modelH, sPSF, exPSF, criterion, opt, type_= 'train', losses = [], epoch= None, m=1, train_model_iter=1, train_H_iter=0, noise=None, noise_K=1):
    losses_temp = []
    opt_model, opt_H = opt
    for i, (x, y) in enumerate(loader):
        #if i==10:break
        x= x.to(device)
        if type_ == 'train':
            model.train()
            modelH.train()

            opt_model.zero_grad()
            opt_H.zero_grad()
            
            X= x.float()
            # train_model
            for _ in range(train_model_iter):
                Ht= modelH(m)
                yt = forward_modelA(X, Ht, sPSF, exPSF, device, noise, noise_K)
                X_hat = model(yt)
                loss = criterion(X_hat, X)
                loss.backward(retain_graph=True)
                opt_model.step()

            #train H
            for _ in range(train_H_iter):
                Ht= modelH(m)
                yt = forward_modelA(X, Ht, sPSF, exPSF, device, noise, noise_K)
                X_hat = model(yt)
                loss = criterion(X_hat, X)
                loss.backward(retain_graph=True)
                
                opt_H.step()
        else:
            model.eval()
            modelH.eval()
            with torch.no_grad():
                X= x.float()
                Ht= modelH(m)
                yt = forward_modelA(X, Ht, sPSF, exPSF, device, noise, noise_K)
                X_hat = model(yt)
                loss = criterion(X_hat, X)
        losses_temp.append(loss.item())
    print(f'yt range ({type_}): [{yt.min()} {yt.max()}]')
    losses.append(np.mean(losses_temp))
    return losses, model, opt, X, X_hat, Ht, yt, modelH

def train(model, forward_modelA, modelH, sPSF, exPSF, criterion, opt, train_loader, test_loader, device, T=5, epochs=100, show_results_epoch=1, train_model_iter=1, train_H_iter=0, m_inc_epoc=1, m_inc_proc= None, save_dir= None, noise_A=None, classifier=None, rescale_for_classifier= [-1, 1], noise_K=1):
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
        
        start = time.time()
        losses_train, model, opt, X, X_hat, Ht, yt, modelH = loop(device, train_loader, model, forward_modelA, modelH, sPSF, exPSF, criterion, opt, 'train', losses_train, epoch, m, train_model_iter, train_H_iter, noise_A, noise_K)
        end= time.time()
        
        losses_test, model, opt, X_val, X_hat_val, Ht_val, yt_val, modelH = loop(device, test_loader, model, forward_modelA, modelH, sPSF, exPSF, criterion, opt, 'test', losses_test, epoch, m, train_model_iter, train_H_iter, noise_A, noise_K)
        
        
        print(f'training loop time (for single epoch): {end-start} sec')
        if classifier!=None:
            class_acc_on_real, class_acc_on_fake = evaluate(device, test_loader, model, forward_modelA, modelH, classifier, sPSF, exPSF, m, noise_A, rescale= rescale_for_classifier, noise_K= noise_K)

        if epoch%show_results_epoch==0:
            if classifier==None:
                acc_on_real, acc_on_fake=None, None
            show_imgs(X_val, Ht_val, X_hat_val, yt_val, losses_train, losses_test, T, epoch, class_acc_on_real, class_acc_on_fake, save_dir)            