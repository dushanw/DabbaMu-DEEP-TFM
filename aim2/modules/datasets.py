import torchvision
import torch
import matplotlib.pyplot as plt
from modules.data_utils import *


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
    data_dir= "/n/home06/udithhaputhanthri/project_udith/datasets/mnistgrid_mnistsize(32)_imgsize(640)"
    
    trainset = mnistgrid_getdataset(img_size, 'train', delta, data_dir, num_samples_train)
    valset= mnistgrid_getdataset(img_size, 'val', delta, data_dir)
    testset= mnistgrid_getdataset(img_size, 'test', delta, data_dir)
    
    return trainset, valset, testset


def confocal(img_size, delta, num_samples_train):
    data_dir= "/n/holyscratch01/wadduwage_lab/uom_Udith/datasets/confocal/63xZseriesSmall_w1/"
    
    trainset = confocal_getdataset(img_size, 'train', delta, data_dir, num_samples_train)
    valset= confocal_getdataset(img_size, 'val', delta, data_dir)
    testset= confocal_getdataset(img_size, 'test', delta, data_dir)
    
    return trainset, valset, testset



def confocal_segment(img_size, delta, num_samples_train):
    data_dir= "/n/holyscratch01/wadduwage_lab/confocal_w1_segmentation_dataset"
    
    trainset = confocal_seg_getdataset(img_size, 'train', delta, data_dir, num_samples_train)
    valset= confocal_seg_getdataset(img_size, 'val', delta, data_dir)
    testset= confocal_seg_getdataset(img_size, 'test', delta, data_dir)
    
    return trainset, valset, testset


def neuronal(img_size, delta, num_samples_train):  
    
    train_data_dir = '/n/holylfs/LABS/wadduwage_lab/Lab/navodini/_results/_cnn_synthTrData/17-Aug-2021/dmd_exp_tfm_mouse_20201224_100um/mouse_Synbv_100um_data_6sls_20mc_tr.h5'
    test_data_dir = '/n/holylfs/LABS/wadduwage_lab/Lab/navodini/_results/_cnn_synthTrData/17-Aug-2021/dmd_exp_tfm_mouse_20201224_100um/mouse_Synbv_100um_data_6sls_20mc_test.h5'
    
    
    trainset = neuronal_getdataset(img_size, 'train', delta, train_data_dir, num_samples_train)
    valset= neuronal_getdataset(img_size, 'val', delta, test_data_dir)
    testset= neuronal_getdataset(img_size, 'test', delta, test_data_dir)
    
    return trainset, valset, testset


def vascular_v1(img_size, delta, num_samples_train): ## murats dataset
    data_dir= "/n/holylfs/LABS/wadduwage_lab/Lab/uom_Udith/datasets/vascular"
    
    trainset = vascular_v1_getdataset(img_size, 'train', delta, data_dir, num_samples_train)
    valset= vascular_v1_getdataset(img_size, 'val', delta, data_dir)
    testset= vascular_v1_getdataset(img_size, 'test', delta, data_dir)
    
    return trainset, valset, testset


