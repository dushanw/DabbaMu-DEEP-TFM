import torch
from torch.nn import functional as F

def conv_HPF(image_batch): #image_batch: (batch_size, 1, img_size, img_size)
    device= image_batch.device
    weights= torch.tensor([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]).view(1,1,3,3).float()/9.0

    image_batch = F.conv2d(image_batch, weights.to(device), padding=1)
    return image_batch

def conv_LPF(image_batch): #image_batch: (batch_size, 1, img_size, img_size)
    device= image_batch.device
    weights= torch.tensor([[1,1,1],[1,1,1],[1,1,1]]).view(1,1,3,3).float()/ 9.0
    image_batch = F.conv2d(image_batch, weights.to(device), padding=1)
    return image_batch

############################################################
def HPF(image_batch): #image_batch: (batch_size, 1, img_size, img_size)
    return conv_HPF(image_batch)

def LPF(image_batch): #image_batch: (batch_size, 1, img_size, img_size)
    return conv_LPF(image_batch)