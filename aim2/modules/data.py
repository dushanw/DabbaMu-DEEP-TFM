import torch
import matplotlib.pyplot as plt
import torchvision
import numpy as np
from PIL import Image
import glob

class mnistgrid_getdataset(torch.utils.data.Dataset):
    def __init__(self, img_size= 32, type_= 'train', delta=0, img_dir= "/n/home06/udithhaputhanthri/project_udith/datasets/mnistgrid_imgsize(32)"):
        super(mnistgrid_getdataset, self).__init__()
        
        self.img_list = glob.glob(f"{img_dir}/{type_}/*.jpg")
        
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
        return self.transform(Image.fromarray(plt.imread(self.img_list[idx]))), torch.tensor(1) # 2nd output-> to maintain consistency across all MNIST dataloaders