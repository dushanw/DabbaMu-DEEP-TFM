import torch 
import torchvision
import matplotlib.pyplot as plt


def show_results(x_fake, x_real, epoch, losses_train, losses_test):
    img_grid_fake=torchvision.utils.make_grid(x_fake*0.5+0.5).permute(1,2,0).cpu().detach()
    img_grid_real=torchvision.utils.make_grid(x_real*0.5+0.5).permute(1,2,0).cpu().detach()

    plt.figure(figsize=(20,3)) #(length, height)
    plt.subplot(1,2,1)
    plt.imshow(img_grid_real)
    plt.title('real')
    plt.subplot(1,2,2)
    plt.imshow(img_grid_fake)
    plt.title('fake')
    plt.suptitle(f'results after {epoch} epochs ... ')
    #plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(losses_train, label= 'train loss')
    plt.plot(losses_test, label= 'test loss')
    plt.legend()
    plt.title(f'losses after {epoch} epochs ... (final test loss : {losses_test[-1]})')
    plt.show()
    
def freeze_encoder(model, freeze= True):
    print(f'freeze Encoder : {freeze}')
    if freeze== True:
        for param in model.LinearEncoder.parameters():
            param.requires_grad= False
    else:
        for param in model.LinearEncoder.parameters():
            param.requires_grad= True
    return model

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)