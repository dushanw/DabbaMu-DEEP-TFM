import scipy.io
import os
import torch

def confocal_w1_exPSF(psf_dir= '../psfs'):
    exPSF = scipy.io.loadmat(f'{psf_dir}/psf_640.mat')['psf_640']
    return torch.from_numpy(exPSF).float()

def confocal_w1_emPSF(psf_dir= '../psfs'):
    emPSF = scipy.io.loadmat(f'{psf_dir}/psf_700.mat')['psf_700']
    return torch.from_numpy(emPSF).float()