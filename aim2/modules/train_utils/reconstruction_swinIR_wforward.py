
from modules.utils import *
from modules.custom_activations import *
import numpy as np
import time

from modules.eval_metrics import ssim_ignite, mse_distance, l1_distance
from modules.models.classifiers import classification_accuracy as accuracy

def loop(device, loader, model_decoder, model_decoder_upsample, model_A, model_H, criterion, opt, type_= 'train', losses = [], epoch= None, m=1, train_model_iter=1, train_H_iter=0, metrics = None, connect_forward_inverse= None):
    
    if connect_forward_inverse== None:connect_forward_inverse= no_skips
    
    losses_temp = []
    metric_ssim11_temp = []
    metric_ssim5_temp = []
    metric_mse_temp = []
    metric_l1_temp = []
    
    opt_model, opt_H = opt
    len_dataset= len(loader.dataset)
    
    for i, (x, y) in enumerate(loader):
        current_step= epoch*len_dataset + i
        #if i==10:break
        x= x.to(device)
        if type_ == 'train':            
            X= x.float()
            # train_model
            model_decoder.update_learning_rate(current_step)
            model_decoder.feed_data({'m':m, 'H':X})
            model_decoder.optimize_parameters(current_step, opt_H) #optimize both model_decoder and Ht

            X_hat = model_decoder.E
            Ht= model_decoder.Ht
            yt_down= model_decoder.yt_down
            
            loss = model_decoder.log_G_loss
        else:
            model_H.eval()
            with torch.no_grad():
                X= x.float()                
                model_decoder.feed_data({'m':m, 'H':X})
                model_decoder.test()
                X_hat = model_decoder.E
                Ht= model_decoder.Ht
                yt_down= model_decoder.yt_down
                
                loss = model_decoder.log_G_loss
        losses_temp.append(loss.item())
        
        metric_ssim11_temp.append(ssim_ignite(X_hat, X, k=11))
        metric_ssim5_temp.append(ssim_ignite(X_hat, X, k=5))
        metric_mse_temp.append(mse_distance(X_hat, X))
        metric_l1_temp.append(l1_distance(X_hat, X))
        
    
    print(f'after {epoch} epochs... yt_down range ({type_}): [{yt_down.min()} {yt_down.max()}]')
    losses.append(np.mean(losses_temp))
    
    metrics['ssim11'].append(np.mean(metric_ssim11_temp))
    metrics['ssim5'].append(np.mean(metric_ssim5_temp))
    metrics['mse'].append(np.mean(metric_mse_temp))
    metrics['l1'].append(np.mean(metric_l1_temp))
    
    return losses, model_decoder, opt, X, X_hat, Ht, yt_down, model_H, metrics

def train(model_decoder, model_decoder_upsample, model_A, model_H, connect_forward_inverse, criterion, opt, train_loader, test_loader, device, epochs=100, show_results_epoch=1, train_model_iter=1, train_H_iter=0, m_inc_proc= None, save_dir= None, classifier=None, rescale_for_classifier= [-1, 1], save_special_bool=False, cfg= None, save_dir_special= None):
    
    T= model_H.T
    
    if m_inc_proc==None:
        def m_inc_proc(m, epoch):
            return m

    losses_train= []
    losses_val= []
    
    metrics = {'l1':[], 'mse':[], 'ssim5':[], 'ssim11':[]}
    metrics_val = {'l1':[], 'mse':[], 'ssim5':[], 'ssim11':[]}
    print(f'device : {device}')

    m=1
    for epoch in range(1, epochs+1):
        m= m_inc_proc(m, epoch)
        print(f'm : {m}')
        
        start_train = time.time()
        losses_train, model_decoder, opt, X, X_hat, Ht, yt_down, model_H, metrics = loop(device, train_loader, model_decoder, model_decoder_upsample, model_A, model_H, criterion, opt, 'train', losses_train, epoch, m, train_model_iter, train_H_iter, metrics, connect_forward_inverse)
        end_train= time.time()
        
        start_val = time.time()
        losses_val, model_decoder, opt, X_val, X_hat_val, Ht_val, yt_down_val, model_H, metrics_val = loop(device, test_loader, model_decoder, model_decoder_upsample, model_A, model_H, criterion, opt, 'test', losses_val, epoch, m, train_model_iter, train_H_iter, metrics_val, connect_forward_inverse)
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
            show_imgs(X_val, Ht_val, X_hat_val, yt_down_val, losses_train, losses_val, metrics_val, T, epoch, class_acc_on_real, class_acc_on_fake, save_dir, m) 
            
        if epoch== epochs or epoch== 1:
            print(f'special saving settings : {save_special_bool} | {save_dir_special}')
            if save_special_bool==True and save_dir_special != None:
                print('special saving started !!!')
                
                save_special_dir= f'{save_dir_special}/save_special'
            
                try:os.mkdir(save_special_dir)
                except:pass
            
                print('special save dir : ', save_special_dir)
                
                save_special(X_val, Ht_val, X_hat_val, yt_down_val, epoch, save_special_dir)                
                
                torch.save({
                    'cfg':cfg,
                    'epoch': epoch,
                    'm':m, 

                    'decoder': model_decoder.state_dict(),
                    #'decoder_upsample': model_decoder_upsample.state_dict(),
                    #'model_H': model_H.state_dict(),
                    
                    'losses_train': losses_train,
                    'losses_val': losses_val,
                    'metrics_train': metrics,
                    'metrics_val': metrics_val,
                    
                    #'opt_decoder_state_dict': opt[0].state_dict(),
                    #'opt_Ht_state_dict': opt[1].state_dict(),
                    }, f'{save_special_dir}/latest_model.pth')
                
            else:
                print('special saving not started !!!')
        
            
            