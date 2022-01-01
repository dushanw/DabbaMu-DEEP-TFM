
from modules.utils import *
from modules.custom_activations import *
import numpy as np
import time
from torch import nn


from modules.eval_metrics import ssim_ignite, mse_distance
from modules.models.classifiers import classification_accuracy as accuracy

def loop(device, loader, model_decoder, model_decoder_upsample, model_A, model_H, criterion, opt, type_= 'train', losses = [], epoch= None, m=1, train_model_iter=1, train_H_iter=0, metrics = None, connect_forward_inverse= None):
    
    if connect_forward_inverse== None:connect_forward_inverse= no_skips
    
    losses_temp = []
    metric_acc_temp = []
    metric_ce_temp = []
    
    opt_model, opt_H = opt
    for i, (x, y) in enumerate(loader):
        #if i==10:break
        x= x.to(device)
        y= y.to(device)
        if type_ == 'train':
            model_decoder.train()
            model_H.train()

            opt_model.zero_grad()
            opt_H.zero_grad()
            
            X= x.float()
            # train_model
            for _ in range(train_model_iter):
                Ht= model_H(m)
                
                lambda_up, yt_down = model_A.compute_yt(X, Ht)
                yt_up = model_decoder_upsample(yt_down, Ht= Ht)
                y_pred = model_decoder(yt_up) # Ht will be used if Ht should be updated through decoder, Therefore depend on the decoder architecture
                loss = criterion(y_pred, y)
                loss.backward(retain_graph=True)
                opt_model.step()

            #train H
            for _ in range(train_H_iter):
                Ht= model_H(m)
                
                lambda_up, yt_down = model_A.compute_yt(X, Ht)
                yt_up = model_decoder_upsample(yt_down, Ht= Ht)
                y_pred = model_decoder(yt_up) # Ht will be used if Ht should be updated through decoder, Therefore depend on the decoder architecture
                
                loss = criterion(y_pred, y)
                loss.backward(retain_graph=True)
                
                opt_H.step()
        else:
            model_decoder.eval()
            model_H.eval()
            with torch.no_grad():
                X= x.float()
                Ht= model_H(m)
                
                lambda_up, yt_down = model_A.compute_yt(X, Ht)
                yt_up = model_decoder_upsample(yt_down, Ht= Ht)
                y_pred = model_decoder(yt_up) # Ht will be used if Ht should be updated through decoder, Therefore depend on the decoder architecture
                
                loss = criterion(y_pred, y)
        losses_temp.append(loss.item())
        
        metric_acc_temp.append(accuracy(y_pred, y))
        metric_ce_temp.append(loss.item())
        
    
    print(f'after {epoch} epochs... yt_down range ({type_}): [{yt_down.min()} {yt_down.max()}]')
    losses.append(np.mean(losses_temp))
    
    metrics['acc'].append(np.mean(metric_acc_temp))
    metrics['ce'].append(np.mean(metric_ce_temp))
    
    return losses, model_decoder, opt, Ht, yt_down, model_H, metrics

def train(model_decoder, model_decoder_upsample, model_A, model_H, connect_forward_inverse, criterion, opt, train_loader, test_loader, device, epochs=100, show_results_epoch=1, train_model_iter=1, train_H_iter=0, m_inc_proc= None, save_dir= None, classifier=None, rescale_for_classifier= [-1, 1], save_special_bool=False, cfg= None, save_dir_special= None):
    
    T= model_H.T
    criterion = nn.CrossEntropyLoss()
    
    if m_inc_proc==None:
        def m_inc_proc(m, epoch):
            return m

    losses_train= []
    losses_val= []
    
    metrics = {'acc':[], 'ce':[]}
    metrics_val = {'acc':[], 'ce':[]}
    print(f'device : {device}')

    m=1
    for epoch in range(1, epochs+1):
        m= m_inc_proc(m, epoch)
        print(f'm : {m}')
        
        start_train = time.time()
        losses_train, model_decoder, opt, Ht, yt, model_H, metrics = loop(device, train_loader, model_decoder, model_decoder_upsample, model_A, model_H, criterion, opt, 'train', losses_train, epoch, m, train_model_iter, train_H_iter, metrics, connect_forward_inverse)
        end_train= time.time()
        
        start_val = time.time()
        losses_val, model_decoder, opt, Ht_val, yt_val, model_H, metrics_val = loop(device, test_loader, model_decoder, model_decoder_upsample, model_A, model_H, criterion, opt, 'test', losses_val, epoch, m, train_model_iter, train_H_iter, metrics_val, connect_forward_inverse)
        end_val= time.time()
        
        
        with open(f"{save_dir}/details.txt", 'a') as f:
            timing_details= f'training loop time (for epoch: {epoch}): {end_train-start_train} sec\n'
            timing_details+= f'validation loop time (for epoch: {epoch}): {end_val-start_val} sec\n\n'
            print(timing_details)
            f.write(timing_details)


        if epoch== epochs:
            class_acc_on_real, class_acc_on_fake=None, None
            show_imgs_classification(Ht_val, yt_val, losses_train, losses_val, metrics_val, T, epoch, class_acc_on_real, class_acc_on_fake, save_dir, m) 
            
        if epoch== epochs:
            if save_special_bool==True:
                
                save_special_dir= f'{save_dir_special}/save_special'
            
                try:os.mkdir(save_special_dir)
                except:pass
            
                print(save_special_dir)
                
                save_special_classification(Ht_val, yt_val, epoch, save_special_dir)                
                
                torch.save({
                    'cfg':cfg,
                    'epoch': epoch,
                    'm':m, 

                    'decoder': model_decoder.state_dict(),
                    'decoder_upsample': model_decoder_upsample.state_dict(),
                    'model_H': model_H.state_dict(),
                    
                    'losses_train': losses_train,
                    'losses_val': losses_val,
                    'metrics_train': metrics,
                    'metrics_val': metrics_val,
                    
                    'opt_decoder_state_dict': opt[0].state_dict(),
                    'opt_Ht_state_dict': opt[1].state_dict(),
                    }, f'{save_special_dir}/latest_model.pth')
                
                
        
            
            