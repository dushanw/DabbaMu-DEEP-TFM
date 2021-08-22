import torchvision
import torch
import matplotlib.pyplot as plt
from modules.data_utils import mnistgrid_getdataset


def mnistdigits(img_size, delta, num_samples_train): #num_samples will not be used 
    
    mean =-delta/(1-delta)
    std=1/(1-delta)
    
    data_dir= '/n/home06/udithhaputhanthri/project_udith/datasets/mnist'
    trainset= torchvision.datasets.MNIST(data_dir, train=True, download=True,
                                 transform=torchvision.transforms.Compose([
                                    torchvision.transforms.Resize([img_size, img_size]),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                     (mean,), (std,))
                                 ]))

    valtestset = torchvision.datasets.MNIST(data_dir, train=False, download=True,
                                 transform=torchvision.transforms.Compose([
                                    torchvision.transforms.Resize([img_size, img_size]),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                     (mean,), (std,))
                                 ]))
    
    valset = valtestset
    testset = None ## create this if needed
    
    return trainset, valset, testset

def mnistdigits_grid2patch(img_size, delta, num_samples_train):
    data_dir= "/n/home06/udithhaputhanthri/project_udith/datasets/mnistgrid_imgsize(32)_v2"
    
    trainset = mnistgrid_getdataset(img_size, 'train', delta, data_dir, num_samples_train)
    valset= mnistgrid_getdataset(img_size, 'val', delta, data_dir)
    testset= mnistgrid_getdataset(img_size, 'test', delta, data_dir)
    
    return trainset, valset, testset