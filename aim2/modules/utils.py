
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import os

import glob
import numpy as np
import torch


def concat_imgs(save_dir, epoch, class_acc_on_fake=None, L1loss=None):
    im_long = plt.imread(f'{save_dir}/{epoch}_line6_results_grids.jpg')
    imgs=[]
    max_width=0
    cummulative_heights=[0]
    widths=[]
    img_dir_list= glob.glob(f'{save_dir}/{epoch}_*')
    for img_dir in sorted(img_dir_list):
        img = plt.imread(img_dir)
        imgs.append(img)
        height, width = img.shape[0], img.shape[1]
        #plt.imshow(img)
        #plt.title(f'height: {height}, width : {width}')
        #plt.show()
        
        if width>max_width:max_width= width
        cummulative_heights.append(cummulative_heights[-1] + height)
        widths.append(width)
    
    final_img= (255*np.ones((cummulative_heights[-1], max_width, 3))).astype('uint8')
    #print(cummulative_heights)
    for i in range(len(cummulative_heights)-1):
        #if not i==4:img= torch.ones_like(torch.tensor(imgs[i])).numpy()
        #else:img= imgs[i]
        img= imgs[i]
        final_img[cummulative_heights[i]:cummulative_heights[i+1], 0:widths[i]]= img
        
    plt.figure(figsize= (15, 10))
    plt.imshow(final_img)
    plt.title(f'after {epoch} epochs')
    plt.show()
    

    if class_acc_on_fake=='NA':
        save_img_dir= f'{save_dir}/{epoch}_L1Loss({L1loss}).jpg'
    else:
        rounded= np.round(float(class_acc_on_fake), 3)
        save_img_dir= f'{save_dir}/{epoch}_AccOnFake({rounded})_L1Loss({L1loss}).jpg'
    plt.imsave(save_img_dir, final_img)

    for img in glob.glob(f'{save_dir}/{epoch}_line*'):
        os.remove(img)

def show_imgs(X, Ht, X_hat, yt, losses_train, losses_test,T, epoch, class_acc_on_real=None, class_acc_on_fake=None, save_dir=None, m=1):
    #vmin=0
    #vmax=1
    print(f'before normalizaton: show images : X range : [{X.min()}, {X.max()}] | X_hat range : [{X_hat.min()}, {X_hat.max()}]')
    
    X = (X-X.min())/(X.max() - X.min())
    X_hat = torch.clamp((X_hat-X_hat.min())/(X_hat.max() - X_hat.min()), 0, 1)
    
    print(f'after normalization: show images : X range : [{X.min()}, {X.max()}] | X_hat range : [{X_hat.min()}, {X_hat.max()}]')
    
    if T>5:T=5
    if class_acc_on_fake==None or class_acc_on_real==None:class_acc_on_fake, class_acc_on_real = 'NA', 'NA'
    if save_dir!=None:
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
            
    idx=np.random.randint(0, len(X))
    plt.figure(figsize= (6, 3))
    plt.subplot(1,2,1)
    plt.imshow(X[idx,0].detach().cpu().numpy())
    plt.title('real')
    plt.subplot(1,2,2)
    plt.imshow(X_hat[idx,0].detach().cpu().numpy())
    plt.title('generated')
    plt.suptitle(f'm : {m}')
    if save_dir!=None:
        plt.savefig(f'{save_dir}/{epoch}_line1_real_gen.jpg')
        plt.close()
    else:plt.show()
    plt.figure(figsize= (2*T, 2))
    for t in range(T):
        plt.subplot(1,T, t+1)
        plt.imshow(Ht[0, t].detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    plt.suptitle('Ht')
    if save_dir!=None:
        plt.savefig(f'{save_dir}/{epoch}_line2_ht.jpg')
        plt.close()
    else:plt.show()
    plt.figure(figsize= (2*T, 2))
    for t in range(T):
        plt.subplot(1,T, t+1)
        plt.imshow(Ht[0, t].detach().cpu().numpy()> 0.5, cmap='gray', vmin=0, vmax=1)
    plt.suptitle('Ht > 0.5')
    if save_dir!=None:
        plt.savefig(f'{save_dir}/{epoch}_line3_ht_comp.jpg')
        plt.close()
    else:plt.show()
    plt.figure(figsize= (2*T, 2))
    for t in range(T):
        plt.subplot(1,T, t+1)
        plt.imshow(yt[idx, t].detach().cpu().numpy())
    plt.suptitle('yt')
    if save_dir!=None:
        plt.savefig(f'{save_dir}/{epoch}_line4_yt.jpg')
        plt.close()
    else:plt.show()
    
    
    plt.figure()
    plt.plot(losses_train, label= 'train loss')
    plt.plot(losses_test, label= 'test loss')
    plt.legend()
    plt.title(f'losses after {epoch} epochs ...: final loss: {losses_test[-1]}')
    if save_dir!=None:
        plt.savefig(f'{save_dir}/{epoch}_line5_losses.jpg')
        plt.close()
    else:plt.show()
    
    img_grid_fake=torchvision.utils.make_grid(X_hat).permute(1,2,0).cpu().detach().numpy()
    img_grid_real=torchvision.utils.make_grid(X).permute(1,2,0).cpu().detach().numpy()

    plt.figure(figsize=(15,3)) #(length, height)
    plt.subplot(1,2,1)
    plt.imshow(img_grid_real)
    plt.title(f'real -> class_acc : {class_acc_on_real} | L1Loss : {np.round(losses_test[-1], 3)}')
    plt.subplot(1,2,2)
    plt.imshow(img_grid_fake)
    plt.title(f'fake -> class_acc :  {class_acc_on_fake} | L1Loss : {np.round(losses_test[-1], 3)}')
    plt.suptitle(f'results after {epoch} epochs ... ')
    if save_dir!=None:
        plt.savefig(f'{save_dir}/{epoch}_line6_results_grids.jpg')
        plt.close()
    else:plt.show()
    
    if save_dir!=None:
        concat_imgs(save_dir, epoch, class_acc_on_fake, np.round(losses_test[-1], 3))
        
        
def save_special(X_val, Ht_val, X_hat_val, yt_val, epoch, save_dir):
    np.save(f'{save_dir}/{epoch}_X_val.npy', X_val.detach().cpu().numpy())
    np.save(f'{save_dir}/{epoch}_Ht_val.npy', Ht_val.detach().cpu().numpy())
    np.save(f'{save_dir}/{epoch}_X_hat_val.npy', X_hat_val.detach().cpu().numpy())
    np.save(f'{save_dir}/{epoch}_yt_val.npy', yt_val.detach().cpu().numpy())