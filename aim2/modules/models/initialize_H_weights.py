from scipy.linalg import hadamard
import numpy as np
import torch

def canonical(T, img_size, lambda_scale_factor): # returns {0, 1} torch tensor with shape: (1, T, img_size, img_size)
    pass

def hadamard_rescaled(T, img_size, lambda_scale_factor):  # returns {0, 1} torch tensor with shape: (1, T, img_size, img_size)
    side_scale_ratio= 2**(lambda_scale_factor-1)
    hadamard_size = side_scale_ratio**2
    
    if hadamard_size<T:
        print("WARNING :: Compression Ratio > 1.0 (no compression), Hadamard only works for Compression Ratio <= 1.0. Initialized Ht weights using : randn_FourierBased (instead of hadamard_FourierBased)")
        
        return torch.randn((1, T, img_size, img_size), dtype= torch.float32)

    hadamard_matrix= hadamard(hadamard_size)

    output= np.zeros((1, T, img_size, img_size))

    for t in range(T):
        hadamard_patch = np.resize(hadamard_matrix[t], (side_scale_ratio, side_scale_ratio))*0.5+0.5 #{0,1}
        output[0, t] = np.tile(hadamard_patch, (img_size//side_scale_ratio,img_size//side_scale_ratio))
        
    return torch.tensor(output, dtype= torch.float32)