import torchvision
import torch
import matplotlib.pyplot as plt
from modules.data_utils import *
import os

def mnistdigits(img_size, delta, num_samples_train, num_samples_valtest=None): #num_samples will not be used 
    
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

def mnistdigits_grid2patch(img_size, delta, num_samples_train, num_samples_valtest=None):
    data_dir= "/n/home06/udithhaputhanthri/project_udith/datasets/mnistgrid_mnistsize(32)_imgsize(640)"
    
    trainset = mnistgrid_getdataset(img_size, 'train', delta, data_dir, num_samples_train)
    valset= mnistgrid_getdataset(img_size, 'val', delta, data_dir)
    testset= mnistgrid_getdataset(img_size, 'test', delta, data_dir)
    
    return trainset, valset, testset


def confocal(img_size, delta, num_samples_train, num_samples_valtest=None):
    data_dir= "/n/holylfs/LABS/wadduwage_lab/Lab/uom_Udith/datasets/confocal/63xZseriesSmall_w1/"
    
    trainset = confocal_getdataset(img_size, 'train', delta, data_dir, num_samples_train)
    valset= confocal_getdataset(img_size, 'val', delta, data_dir)
    testset= confocal_getdataset(img_size, 'test', delta, data_dir)
    
    return trainset, valset, testset



def confocal_segment(img_size, delta, num_samples_train, num_samples_valtest=None):
    #data_dir= "/n/holyscratch01/wadduwage_lab/confocal_w1_segmentation_dataset"
    data_dir= "/n/holylfs/LABS/wadduwage_lab/Lab/uom_Udith/datasets/confocal_w1_segmentation_dataset"
    
    trainset = confocal_seg_getdataset(img_size, 'train', delta, data_dir, num_samples_train)
    valset= confocal_seg_getdataset(img_size, 'val', delta, data_dir)
    testset= confocal_seg_getdataset(img_size, 'test', delta, data_dir)
    
    return trainset, valset, testset


def neuronal(img_size, delta, num_samples_train, num_samples_valtest=None):  
    
    train_data_dir = '/n/holylfs/LABS/wadduwage_lab/Lab/navodini/_results/_cnn_synthTrData/17-Aug-2021/dmd_exp_tfm_mouse_20201224_100um/mouse_Synbv_100um_data_6sls_20mc_tr.h5'
    test_data_dir = '/n/holylfs/LABS/wadduwage_lab/Lab/navodini/_results/_cnn_synthTrData/17-Aug-2021/dmd_exp_tfm_mouse_20201224_100um/mouse_Synbv_100um_data_6sls_20mc_test.h5'
    
    
    trainset = neuronal_getdataset(img_size, 'train', delta, train_data_dir, num_samples_train)
    valset= neuronal_getdataset(img_size, 'val', delta, test_data_dir)
    testset= neuronal_getdataset(img_size, 'test', delta, test_data_dir)
    
    return trainset, valset, testset


def vascular_v1(img_size, delta, num_samples_train, num_samples_valtest=None): ## murats dataset
    data_dir= "/n/holylfs/LABS/wadduwage_lab/Lab/uom_Udith/datasets/vascular"
    
    trainset = vascular_v1_getdataset(img_size, 'train', delta, data_dir, num_samples_train)
    valset= vascular_v1_getdataset(img_size, 'val', delta, data_dir)
    testset= vascular_v1_getdataset(img_size, 'test', delta, data_dir)
    
    return trainset, valset, testset


def bbbcHumanMCF7cellsW2(img_size, delta, num_samples_train, num_samples_valtest):
    data_dir= '/n/holylfs/LABS/wadduwage_lab/Lab/uom_Udith/datasets/bbbcHumanMCF7cells/preprocessed/w2'
    
    trainset = bbbcHumanMCF7cellsW2_getdataset(img_size, 'train', delta, data_dir, num_samples_train)
    valset= bbbcHumanMCF7cellsW2_getdataset(img_size, 'val', delta, data_dir, num_samples_valtest)
    testset= bbbcHumanMCF7cellsW2_getdataset(img_size, 'test', delta, data_dir, num_samples_valtest)
    
    return trainset, valset, testset


def bbbcHumanMCF7cellsW4(img_size, delta, num_samples_train, num_samples_valtest):

    data_dir= '/n/holylfs/LABS/wadduwage_lab/Lab/uom_Udith/datasets/bbbcHumanMCF7cells/preprocessed/w4'
    if not os.path.isdir(data_dir):
        data_dir= '/home/udith/udith_works/datasets/bbbcHumanMCF7cells/preprocessed/w4' # handle lab server

    trainset = bbbcHumanMCF7cellsW4_getdataset(img_size, 'train', delta, data_dir, num_samples_train)
    valset= bbbcHumanMCF7cellsW4_getdataset(img_size, 'val', delta, data_dir, num_samples_valtest)
    testset= bbbcHumanMCF7cellsW4_getdataset(img_size, 'test', delta, data_dir, num_samples_valtest)
    
    return trainset, valset, testset

def bbbcHumanMCF7cellsW4cropped(img_size, delta, num_samples_train, num_samples_valtest):

    data_dir= '/n/holylfs/LABS/wadduwage_lab/Lab/uom_Udith/datasets/bbbcHumanMCF7cells/preprocessed/w4'
    if not os.path.isdir(data_dir):
        data_dir= '/home/udith/udith_works/datasets/bbbcHumanMCF7cells/preprocessed/w4' # handle lab server

    trainset = bbbcHumanMCF7cellsW4_getdataset(img_size, 'train', delta, data_dir, num_samples_train, is_crop= True)
    valset= bbbcHumanMCF7cellsW4_getdataset(img_size, 'val', delta, data_dir, num_samples_valtest, is_crop= True)
    testset= bbbcHumanMCF7cellsW4_getdataset(img_size, 'test', delta, data_dir, num_samples_valtest, is_crop= True)
    
    return trainset, valset, testset


def div2kflickr2k(img_size, delta, num_samples_train, num_samples_valtest):
    data_dir= '../../../datasets/superres'
    if not os.path.isdir(data_dir):
        data_dir= '/n/holylfs/LABS/wadduwage_lab/Users/udithhaputhanthri/super_resolution_datasets'
    
    data_div2k_train_dir = f'{data_dir}/div2k/DIV2K_train_HR'
    data_div2k_val_dir = f'{data_dir}/div2k/DIV2K_valid_HR'
    data_flickr_dir = f'{data_dir}/flickr2k/Flickr2K/Flickr2K_HR'

    trainset = div2kflickr2k_getdataset(img_size, 'train', delta, data_div2k_train_dir, data_flickr_dir, num_samples_train)
    valset= div2kflickr2k_getdataset(img_size, 'val', delta, data_div2k_val_dir, None, num_samples_valtest)
    testset= valset
    
    return trainset, valset, testset


def bloodvesselsDeepTFM6sls(img_size, delta, num_samples_train, num_samples_valtest):

    #data_dir= '/n/holylfs/LABS/wadduwage_lab/Lab/uom_Udith/datasets/bloodvesselsDeepTFM6sls'
    data_dir= '/n/holyscratch01/wadduwage_lab/uom_Udith/datasets/bloodvesselsDeepTFM6sls'
    if not os.path.isdir(data_dir):
        data_dir= 'add'
    
    trainset = bloodvesselsDeepTFM6sls_getdataset(img_size, 'train', delta, data_dir, num_samples_train, is_crop= True)
    valset= bloodvesselsDeepTFM6sls_getdataset(img_size, 'val', delta, data_dir, num_samples_valtest, is_crop= True)
    testset= bloodvesselsDeepTFM6sls_getdataset(img_size, 'test', delta, data_dir, num_samples_valtest, is_crop= True)
    
    return trainset, valset, testset

