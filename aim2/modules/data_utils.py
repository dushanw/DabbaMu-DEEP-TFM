import torch
import matplotlib.pyplot as plt
import torchvision
import numpy as np
from PIL import Image
import glob
import random
import h5py
import random



class mnistgrid_getdataset(torch.utils.data.Dataset):
    def __init__(self, img_size= 32, type_= 'train', delta=0, img_dir= None, num_samples= None):
        super(mnistgrid_getdataset, self).__init__()
        
        self.type_ = type_
        img_list = sorted(glob.glob(f"{img_dir}/{self.type_}/*.jpg"), key= lambda x: int(x.split('/')[-1][:-4]))
        print(f'total images found in: {img_dir}/{self.type_} -> {len(img_list)}')
        
        if num_samples==None:num_samples=len(img_list)
            
        if len(img_list)<num_samples:
            print(f'WARNING -> Dataset: len(images) < num_samples -> num_samples will be neglected !!!')
            self.img_list= img_list
        else:
            self.img_list= img_list[:num_samples]
        
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
        
        try:output = self.transform(Image.fromarray(plt.imread(self.img_list[idx]))), torch.tensor(1) # 2nd output-> to maintain consistency across all MNIST dataloaders
        except:print('error : ', plt.imread(self.img_list[idx]).shape, self.img_list[idx])
        torch.manual_seed(np.random.randint(0, 500000))
        return output
    
    
class confocal_getdataset(torch.utils.data.Dataset):
    def __init__(self, img_size= 32, type_= 'train', delta=0, img_dir= None, num_samples= None):
        super(confocal_getdataset, self).__init__()
        
        self.type_ = type_
        
        img_list = sorted(glob.glob(f"{img_dir}/{self.type_}/*.png"), key= self.sort_by_stk_and_patch_idx)
        print(f'total images found in: {img_dir}/{self.type_} -> {len(img_list)}')
        
        if num_samples==None:num_samples=len(img_list)
            
        if len(img_list)<num_samples:
            print(f'WARNING -> Dataset: len(images) < num_samples -> num_samples will be neglected !!!')
            self.img_list= img_list
        else:
            self.img_list= img_list[:num_samples]
        
        self.delta= delta
        
        self.mean =-self.delta/(1-self.delta)
        self.std=1/(1-self.delta)
        
        self.transform = torchvision.transforms.Compose([
                                    torchvision.transforms.Resize([img_size, img_size]),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize((self.mean,), (self.std,))])
        
    def sort_by_stk_and_patch_idx(self, x):
        stk_idx = int(x.split('/')[-1][:-4].split('_')[-2][1:])
        patch_idx = int(x.split('/')[-1][:-4].split('_')[-1])
        return stk_idx, patch_idx
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        output = self.transform(Image.fromarray((255*plt.imread(self.img_list[idx])).astype('uint8'))), torch.tensor(1) 
        return output
    
    
    

class confocal_seg_getdataset(torch.utils.data.Dataset):
    def __init__(self, img_size= 32, type_= 'train', delta=0, img_dir= None, num_samples= None):
        super(confocal_seg_getdataset, self).__init__()
        
        self.type_ = type_
        
        img_list = sorted(glob.glob(f"{img_dir}/{self.type_}/images/*.png"), key= self.sort_by_stk_and_patch_idx)
        segmap_list = sorted(glob.glob(f"{img_dir}/{self.type_}/seg_maps/*.png"), key= self.sort_by_stk_and_patch_idx)

        print(f'total images found in: {img_dir}/{self.type_} -> {len(img_list)}')
        
        if num_samples==None:num_samples=len(img_list)
            
        if len(img_list)<num_samples:
            print(f'WARNING -> Dataset: len(images) < num_samples -> num_samples will be neglected !!!')
            self.img_list= img_list
            self.segmap_list= segmap_list
        else:
            self.img_list= img_list[:num_samples]
            self.segmap_list= segmap_list[:num_samples]
        
        self.delta= delta
        self.spot_thresh= 0.1
        
        self.mean =-self.delta/(1-self.delta)
        self.std=1/(1-self.delta)
        
        self.transform = torchvision.transforms.Compose([
                                    torchvision.transforms.Resize([img_size, img_size]),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize((self.mean,), (self.std,))])
        
    def sort_by_stk_and_patch_idx(self, x):
        stk_idx = int(x.split('/')[-1][:-4].split('_')[-2][1:])
        patch_idx = int(x.split('/')[-1][:-4].split('_')[-1])
        return stk_idx, patch_idx
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img= self.transform(Image.fromarray((255*plt.imread(self.img_list[idx])).astype('uint8')))
        segmap= self.transform(Image.fromarray((255*(plt.imread(self.segmap_list[idx]) > self.spot_thresh).astype('float')).astype('uint8')))
        output = img, segmap
        return output
    

class neuronal_getdataset(torch.utils.data.Dataset):
    def __init__(self, img_size= 32, type_= 'train', delta=0, img_dir= None, num_samples= None):
        super(neuronal_getdataset, self).__init__()
        
        self.type_ = type_
        
        f = h5py.File(img_dir, 'r')
        img_set = f['gt']
        print(f'total images found in: [{type_}] -- {img_dir} -> {len(img_set)}')
        
        if num_samples==None:num_samples=len(img_set)
            
        if len(img_set)<num_samples:
            print(f'WARNING -> Dataset: len(images) < num_samples -> num_samples will be neglected !!!')
            self.img_set= img_set
        else:
            self.img_set= img_set[:num_samples]
        
        self.delta= delta
        
        self.mean =-self.delta/(1-self.delta)
        self.std=1/(1-self.delta)
        
        self.transform = torchvision.transforms.Compose([
                                    torchvision.transforms.Resize([img_size, img_size]),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize((self.mean,), (self.std,))])
    def __len__(self):
        return len(self.img_set)
    
    def __getitem__(self, idx):
        output = self.transform(Image.fromarray((255*self.img_set[idx, 0]).astype('uint8'))), torch.tensor(1) 
        return output
    
class vascular_v1_getdataset(torch.utils.data.Dataset):
    def __init__(self, img_size= 32, type_= 'train', delta=0, img_dir= None, num_samples= None):
        super(vascular_v1_getdataset, self).__init__()
        
        self.type_ = type_
        
        img_list = sorted(glob.glob(f"{img_dir}/{self.type_}/*.png"))
        print(f'total images found in: {img_dir}/{self.type_} -> {len(img_list)}')
        
        if num_samples==None:num_samples=len(img_list)
            
        if len(img_list)<num_samples:
            print(f'WARNING -> Dataset: len(images) < num_samples -> num_samples will be neglected !!!')
            self.img_list= img_list
        else:
            self.img_list= img_list[:num_samples]
        
        self.delta= delta
        
        self.mean =-self.delta/(1-self.delta)
        self.std=1/(1-self.delta)
        
        if self.type_ == 'train':
            self.transform = torchvision.transforms.Compose([
                                    torchvision.transforms.Resize([img_size, img_size]),
                                    torchvision.transforms.RandomResizedCrop(size= img_size, scale=(0.5, 1.0), ratio=(0.75, 1.33)),
                                    torchvision.transforms.RandomHorizontalFlip(),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize((self.mean,), (self.std,))])
        else:
            self.transform = torchvision.transforms.Compose([
                                    torchvision.transforms.Resize([img_size, img_size]),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize((self.mean,), (self.std,))])
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        output = self.transform(Image.fromarray((255*plt.imread(self.img_list[idx])).astype('uint8'))), torch.tensor(1) 
        return output
    
    

class bbbcHumanMCF7cellsW2_getdataset(torch.utils.data.Dataset):
    def __init__(self, img_size= 32, type_= 'train', delta=0, img_dir= None, num_samples= None):
        super(bbbcHumanMCF7cellsW2_getdataset, self).__init__()
        
        self.type_ = type_
        
        img_list = sorted(glob.glob(f"{img_dir}/{self.type_}/*.png"))
        print(f'total images found in: {img_dir}/{self.type_} -> {len(img_list)}')
        
        np.random.seed(10)
        np.random.shuffle(img_list)
        
        if num_samples==None:num_samples=len(img_list)
            
        if len(img_list)<num_samples:
            print(f'WARNING -> Dataset: len(images) < num_samples -> num_samples will be neglected !!!')
            self.img_list= img_list
        else:
            self.img_list= img_list[:num_samples]
        
        self.delta= delta
        
        self.mean =-self.delta/(1-self.delta)
        self.std=1/(1-self.delta)
        
        self.transform = torchvision.transforms.Compose([
                                    torchvision.transforms.Resize([img_size, img_size]),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize((self.mean,), (self.std,))])
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        # clip from 0.5 and do min-max norm
        output = self.transform(Image.fromarray((255*plt.imread(self.img_list[idx])).astype('uint8'))), torch.tensor(1) 
        return output
    

class bbbcHumanMCF7cellsW4_getdataset(torch.utils.data.Dataset):
    def __init__(self, img_size= 32, type_= 'train', delta=0, img_dir= None, num_samples= None):
        super(bbbcHumanMCF7cellsW4_getdataset, self).__init__()
        
        self.type_ = type_
        
        img_list = sorted(glob.glob(f"{img_dir}/{self.type_}/*.png"))
        print(f'total images found in: {img_dir}/{self.type_} -> {len(img_list)}')
        
        np.random.seed(10)
        np.random.shuffle(img_list)
        
        if num_samples==None:num_samples=len(img_list)
            
        if len(img_list)<num_samples:
            print(f'WARNING -> Dataset: len(images) < num_samples -> num_samples will be neglected !!!')
            self.img_list= img_list
        else:
            self.img_list= img_list[:num_samples]
        
        self.delta= delta
        
        self.mean =-self.delta/(1-self.delta)
        self.std=1/(1-self.delta)
        
        self.transform = torchvision.transforms.Compose([
                                    torchvision.transforms.Resize([img_size, img_size]),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize((self.mean,), (self.std,))])
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        # clip from 0.5 and do min-max norm
        output = self.transform(Image.fromarray((255*plt.imread(self.img_list[idx])).astype('uint8'))), torch.tensor(1) 
        return output
    
    
class div2kflickr2k_getdataset(torch.utils.data.Dataset):
    def __init__(self, img_size= 32, type_= 'train', delta=0, img_dir_div2k= None, img_dir_flickr2k= None, num_samples= None):
        super(div2kflickr2k_getdataset, self).__init__()
        
        self.type_ = type_

        if self.type_== 'train':
            img_list = glob.glob(f"{img_dir_div2k}/*.png") + glob.glob(f"{img_dir_flickr2k}/*.png")
        else:
            img_list = sorted(glob.glob(f"{img_dir_div2k}/*.png"))

        print(f'total images found in: {self.type_} -> {len(img_list)}')
        
        np.random.seed(10)
        np.random.shuffle(img_list)
        
        if num_samples==None:num_samples=len(img_list)
            
        if len(img_list)<num_samples:
            print(f'WARNING -> Dataset: len(images) < num_samples -> num_samples will be neglected !!!')
            self.img_list= img_list
        else:
            self.img_list= img_list[:num_samples]
        
        self.delta= 0
        
        self.mean =-self.delta/(1-self.delta)
        self.std=1/(1-self.delta)
        
        self.transform_train = torchvision.transforms.Compose([
                                    #torchvision.transforms.Resize([img_size, img_size]),
                                    torchvision.transforms.RandomCrop(img_size),
                                    torchvision.transforms.Grayscale(1),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize((self.mean,), (self.std,))])
        self.transform_valtest = torchvision.transforms.Compose([
                                    #torchvision.transforms.Resize([img_size, img_size]),
                                    torchvision.transforms.CenterCrop(img_size),
                                    torchvision.transforms.Grayscale(1),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize((self.mean,), (self.std,))])
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):        
        if self.type_== 'train':
            img= self.transform_train(Image.fromarray((255*plt.imread(self.img_list[idx])).astype('uint8')))
            mode = random.randint(0, 7)
        else:
            img= self.transform_valtest(Image.fromarray((255*plt.imread(self.img_list[idx])).astype('uint8')))
            mode = idx%8
        img= self.augment_img(img.permute(1,2,0), mode=mode).permute(2, 0, 1) ## augmentation is applied for train, val, test datasets -> because our goal: evaluate on other datasets
        output = img, torch.tensor(1) 
        return output

    def augment_img(self, img, mode=0): #img: HWC
        '''Kai Zhang (github: https://github.com/cszn)
        '''
        if mode == 0:
            return img
        elif mode == 1:
            return torch.flipud(torch.rot90(img))
        elif mode == 2:
            return torch.flipud(img)
        elif mode == 3:
            return torch.rot90(img, k=3)
        elif mode == 4:
            return torch.flipud(torch.rot90(img, k=2))
        elif mode == 5:
            return torch.rot90(img)
        elif mode == 6:
            return torch.rot90(img, k=2)
        elif mode == 7:
            return torch.flipud(torch.rot90(img, k=3))
    
    
    
    
    
    
def return_dataloaders(trainset, valset, testset, batch_size_train= 32, drop_last_val_test= False, batch_size_valtest= 25):
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=True, drop_last= True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size_valtest, shuffle=False, drop_last= drop_last_val_test) # batch_sizes fixed
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size_valtest, shuffle=False, drop_last= drop_last_val_test) # batch_sizes fixed

    print(f'dataset lenths : {len(trainset)} | {len(valset)} | {len(testset)}')
    plt.figure()
    x, y= next(iter(val_loader))
    plt.imshow(x[0,0])
    plt.title('sample datapoint : (from val loader)')
    plt.show()
    
    vmin= x.min().item()
    vmax= x.max().item()
    print('dataset value range : ',vmin, vmax)
    
    return train_loader, val_loader, test_loader


