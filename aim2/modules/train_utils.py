
from modules.utils import *
from modules.custom_activations import *
import numpy as np
import time


from modules.eval_metrics import ssim_ignite, mse_distance
from modules.models.classifiers import classification_accuracy as accuracy

def evaluate(device, loader, model_decoder, model_A, model_H, classifier, rescale=[-1, 1]):
    if rescale!=None:rescale_mean , rescale_std = rescale
    else:rescale_mean , rescale_std= 0, 1
    
    acc_on_real=[]
    acc_on_fake=[]
    for i, (x, y) in enumerate(loader):
        x= x.to(device)
        y= y.to(device)
        model_decoder.eval()
        model_H.eval()
        with torch.no_grad():
            X= x.float()
            Ht= model_H(m)
            yt = model_A.compute_yt(X, Ht)
            X_hat = model_decoder(yt, Ht= Ht) # Ht will be used if Ht should be updated through decoder, Therefore depend on the decoder architecture
            
            acc_on_real.append(accuracy(y, classifier(X*rescale_std+rescale_mean)))
            acc_on_fake.append(accuracy(y, classifier(X_hat*rescale_std+rescale_mean)))
            
    return np.mean(acc_on_real), np.mean(acc_on_fake)


def loop(device, loader, model_decoder, model_decoder_upsample, model_A, model_H, criterion, opt, type_= 'train', losses = [], epoch= None, m=1, train_model_iter=1, train_H_iter=0, metrics = None):
    losses_temp = []
    metric_ssim11_temp = []
    metric_ssim5_temp = []
    metric_mse_temp = []
    
    opt_model, opt_H = opt
    for i, (x, y) in enumerate(loader):
        #if i==10:break
        x= x.to(device)
        if type_ == 'train':
            model_decoder.train()
            model_H.train()

            opt_model.zero_grad()
            opt_H.zero_grad()
            
            X= x.float()
            # train_model
            for _ in range(train_model_iter):
                Ht= model_H(m)
                
                yt = model_A.compute_yt(X, Ht)
                yt_up = model_decoder_upsample(yt, Ht= Ht)
                X_hat = model_decoder(yt_up, Ht= Ht) # Ht will be used if Ht should be updated through decoder, Therefore depend on the decoder architecture
                
                loss = criterion(X_hat, X)
                loss.backward(retain_graph=True)
                opt_model.step()

            #train H
            for _ in range(train_H_iter):
                Ht= model_H(m)
                
                yt = model_A.compute_yt(X, Ht)
                yt_up = model_decoder_upsample(yt, Ht= Ht)
                X_hat = model_decoder(yt_up, Ht= Ht) # Ht will be used if Ht should be updated through decoder, Therefore depend on the decoder architecture
                
                loss = criterion(X_hat, X)
                loss.backward(retain_graph=True)
                
                opt_H.step()
        else:
            model_decoder.eval()
            model_H.eval()
            with torch.no_grad():
                X= x.float()
                Ht= model_H(m)
                
                yt = model_A.compute_yt(X, Ht)
                yt_up = model_decoder_upsample(yt, Ht= Ht)
                X_hat = model_decoder(yt_up, Ht= Ht) # Ht will be used if Ht should be updated through decoder, Therefore depend on the decoder architecture
                
                loss = criterion(X_hat, X)
        losses_temp.append(loss.item())
        
        metric_ssim11_temp.append(ssim_ignite(X_hat, X, k=11))
        metric_ssim5_temp.append(ssim_ignite(X_hat, X, k=5))
        metric_mse_temp.append(mse_distance(X_hat, X))
        
    
    print(f'yt range ({type_}): [{yt.min()} {yt.max()}]')
    losses.append(np.mean(losses_temp))
    
    metrics['ssim11'].append(np.mean(metric_ssim11_temp))
    metrics['ssim5'].append(np.mean(metric_ssim5_temp))
    metrics['mse'].append(np.mean(metric_mse_temp))
    
    return losses, model_decoder, opt, X, X_hat, Ht, yt, model_H, metrics

def train(model_decoder, model_decoder_upsample, model_A, model_H, criterion, opt, train_loader, test_loader, device, epochs=100, show_results_epoch=1, train_model_iter=1, train_H_iter=0, m_inc_proc= None, save_dir= None, classifier=None, rescale_for_classifier= [-1, 1], save_special_bool=False):
    
    T= model_H.T
    
    if m_inc_proc==None:
        def m_inc_proc(m, epoch):
            return m

    losses_train= []
    losses_val= []
    
    metrics = {'mse':[], 'ssim5':[], 'ssim11':[]}
    metrics_val = {'mse':[], 'ssim5':[], 'ssim11':[]}
    print(f'device : {device}')

    m=1
    for epoch in range(1, epochs+1):
        m= m_inc_proc(m, epoch)
        print(f'm : {m}')
        
        start_train = time.time()
        losses_train, model_decoder, opt, X, X_hat, Ht, yt, model_H, metrics = loop(device, train_loader, model_decoder, model_decoder_upsample, model_A, model_H, criterion, opt, 'train', losses_train, epoch, m, train_model_iter, train_H_iter, metrics)
        end_train= time.time()
        
        start_val = time.time()
        losses_val, model_decoder, opt, X_val, X_hat_val, Ht_val, yt_val, model_H, metrics_val = loop(device, test_loader, model_decoder, model_decoder_upsample, model_A, model_H, criterion, opt, 'test', losses_val, epoch, m, train_model_iter, train_H_iter, metrics_val)
        end_val= time.time()
        
        with open(f"{save_dir}/details.txt", 'a') as f:
            timing_details= f'training loop time (for epoch: {epoch}): {end_train-start_train} sec\n'
            timing_details+= f'validation loop time (for epoch: {epoch}): {end_val-start_val} sec\n\n'
            print(timing_details)
            f.write(timing_details)
        
        if classifier!=None:
            class_acc_on_real, class_acc_on_fake = evaluate(device, test_loader, model_decoder, model_A, model_H, classifier, m, rescale= rescale_for_classifier)

        if epoch%show_results_epoch==0:
            if classifier==None:
                class_acc_on_real, class_acc_on_fake=None, None
            show_imgs(X_val, Ht_val, X_hat_val, yt_val, losses_train, losses_val, metrics_val, T, epoch, class_acc_on_real, class_acc_on_fake, save_dir, m) 
            
            if save_special_bool==True:
                save_special_dir= f'{save_dir}/save_special'
                try:os.mkdir(save_special_dir)
                except:pass
                
                save_special(X_val, Ht_val, X_hat_val, yt_val, epoch, f'{save_dir}/save_special')
                
        
            
            