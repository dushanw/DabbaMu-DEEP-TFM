import torch
import matplotlib.pyplot as plt
import torchvision
import numpy as np
from PIL import Image
import glob
import random

class mnistgrid_getdataset(torch.utils.data.Dataset):
    def __init__(self, img_size= 32, type_= 'train', delta=0, img_dir= None):
        super(mnistgrid_getdataset, self).__init__()
        
        self.type_ = type_
        self.img_list = glob.glob(f"{img_dir}/{self.type_}/*.jpg")
        
        self.delta= delta
        
        self.mean =-self.delta/(1-self.delta)
        self.std=1/(1-self.delta)
        
        self.transform = transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize((self.mean,), (self.std,)),
                                torchvision.transforms.RandomCrop((img_size,img_size), padding=None, pad_if_needed=False, fill=0, padding_mode='constant'),
                                torchvision.transforms.Grayscale(num_output_channels=1)])
        
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        torch.manual_seed(idx)  # explanation: below
        
        #apply same random crop for image with same idx. What this do is, creating a unique random crop margins for each image in the dataset (can be train/ validation/ test). So it makes this random cropped datasets fixed as normal datasets but preserving the complexity.
        # Note that, if we apply this only to validation and test sets, it results unlimited large training set because for every epoch, it generates entirely new batches. This will not be similar to real situations where we have limited amount of data. So it is better to keep the randomness fixed for each image. 
        
        output = self.transform(Image.fromarray(plt.imread(self.img_list[idx]))), torch.tensor(1) # 2nd output-> to maintain consistency across all MNIST dataloaders
        torch.manual_seed(np.random.randint(0, 500000))
        return output
    
    
    
def return_dataloaders(trainset, valset, testset, batch_size_train= 32):
    global test_loader
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=True, drop_last= True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=25, shuffle=False, drop_last= False) # batch_sizes fixed
    test_loader = torch.utils.data.DataLoader(testset, batch_size=25, shuffle=False, drop_last= False) # batch_sizes fixed

    plt.figure()
    x, y= next(iter(val_loader))
    plt.imshow(x[0,0])
    plt.title('sample datapoint : (from val loader)')
    plt.show()
    
    vmin= x.min().item()
    vmax= x.max().item()
    print('dataset value range : ',vmin, vmax)
    
    return train_loader, val_loader, test_loader