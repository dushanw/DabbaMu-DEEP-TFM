
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import os

import glob
import numpy as np
import torch


def concat_imgs(save_dir, epoch, class_acc_on_fake=None):
    im_long = plt.imread(f'{save_dir}/{epoch}_line6_results_grids.jpg')
    
    final_img=[]
    idx=0
    for img_dir in sorted(glob.glob(f'{save_dir}/{epoch}_*')):
        im= plt.imread(img_dir)
        ones= (255*np.ones((im.shape[0], im_long.shape[1], 3))).astype('uint8')
        if idx==0 or idx==4:ones[:im.shape[0], 30:30+im.shape[1], :]= im
        elif idx==5:ones[:, :-30, :]= im[:, 30:, :]
        else:ones[:im.shape[0], :im.shape[1], :]= im
        final_img.append(ones)
        idx+=1
    final_img = np.concatenate(tuple(final_img), axis=0)

    if class_acc_on_fake==None:
        save_img_dir= f'{save_dir}/{epoch}.jpg'
    else:
        rounded= np.round(float(class_acc_on_fake), 3)
        save_img_dir= f'{save_dir}/{epoch}_AccOnFake({rounded}).jpg'
    plt.imsave(save_img_dir, final_img)

    for img in glob.glob(f'{save_dir}/{epoch}_line*'):
        os.remove(img)

def show_imgs(X, Ht, X_hat, yt, losses_train, losses_test,T, epoch, class_acc_on_real=None, class_acc_on_fake=None, save_dir=None):
    #vmin=0
    #vmax=1
    print(f'show images : X range : [{X.min()}, {X.max()}]')
    print(f'show images : X_hat range : [{X_hat.min()}, {X_hat.max()}]')
    
    X = (X-X.min())/(X.max() - X.min())
    X_hat = torch.clamp((X_hat-X_hat.min())/(X_hat.max() - X_hat.min()), 0, 1)
    
    print('X.min, X.max, X_hat.min, X_hat.max (after normalization): ',X.min(), X.max(), X_hat.min(), X_hat.max())

    
    if T>5:T=5
    if class_acc_on_fake==None or class_acc_on_real==None:class_acc_on_fake, class_acc_on_real = 'not calc', 'not calc'
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
    if save_dir!=None:
        plt.savefig(f'{save_dir}/{epoch}_line1_real_gen.jpg')
    plt.show()
    plt.figure(figsize= (2*T, 2))
    for t in range(T):
        plt.subplot(1,T, t+1)
        plt.imshow(Ht[0, t].detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    plt.suptitle('Ht')
    if save_dir!=None:
        plt.savefig(f'{save_dir}/{epoch}_line2_ht.jpg')
    plt.show()
    plt.figure(figsize= (2*T, 2))
    for t in range(T):
        plt.subplot(1,T, t+1)
        plt.imshow(Ht[0, t].detach().cpu().numpy()> 0.5, cmap='gray', vmin=0, vmax=1)
    plt.suptitle('Ht > 0.5')
    if save_dir!=None:
        plt.savefig(f'{save_dir}/{epoch}_line3_ht_comp.jpg')
    plt.show()
    plt.figure(figsize= (2*T, 2))
    for t in range(T):
        plt.subplot(1,T, t+1)
        plt.imshow(yt[idx, t].detach().cpu().numpy())
    plt.suptitle('yt')
    if save_dir!=None:
        plt.savefig(f'{save_dir}/{epoch}_line4_yt.jpg')
    plt.show()
    

    plt.plot(losses_train, label= 'train loss')
    plt.plot(losses_test, label= 'test loss')
    plt.legend()
    plt.title(f'losses after {epoch} epochs ...: final loss: {losses_test[-1]}')
    if save_dir!=None:
        plt.savefig(f'{save_dir}/{epoch}_line5_losses.jpg')
    plt.show()
    
    img_grid_fake=torchvision.utils.make_grid(X_hat).permute(1,2,0).cpu().detach().numpy()
    img_grid_real=torchvision.utils.make_grid(X).permute(1,2,0).cpu().detach().numpy()

    plt.figure(figsize=(15,3)) #(length, height)
    plt.subplot(1,2,1)
    plt.imshow(img_grid_real)
    plt.title(f'real -> classification_acc : {class_acc_on_real}')
    plt.subplot(1,2,2)
    plt.imshow(img_grid_fake)
    plt.title(f'fake -> classification_acc :  {class_acc_on_fake}')
    plt.suptitle(f'results after {epoch} epochs ... ')
    if save_dir!=None:
        plt.savefig(f'{save_dir}/{epoch}_line6_results_grids.jpg')
    plt.show()
    
    if save_dir!=None:
        concat_imgs(save_dir, epoch, class_acc_on_fake)