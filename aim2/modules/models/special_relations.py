
import torch
import numpy as np

def get_yt_map_idx_grid(img_size, scale_factor):
        yt_img_size= img_size//scale_factor
        yt_idx_grid= torch.arange(yt_img_size**2).reshape(yt_img_size, yt_img_size)
        yt_map_idx_grid_flatten= torch.tile(yt_idx_grid, (scale_factor,scale_factor,1,1)).permute(2, 0, 3, 1).reshape(img_size, img_size).flatten()

        return yt_map_idx_grid_flatten

def convert_Ht2A(Ht, lambda_scale_factor):
        """
        Convert Ht to sparse matrix A/ forward model matrix including downsampling

        Ht.shape: [1, T, img_size, img_size]
        X.shape: [n_imgs, 1, img_size, img_size]
        """
        img_size= Ht.shape[2]
        device= Ht.device
        
        T= Ht.shape[1]
        scale_factor= 2**(lambda_scale_factor-1)
        yt_img_size= img_size//scale_factor

        Ht_flatten= Ht.reshape(-1) #shape: (T*img_size*img_size, )
        A= torch.zeros(T, yt_img_size**2, img_size**2).float().to(device) #shape: (T, yt_img_size**2, img_size**2)  ## CAUTION: MEMORY HUNGRY- 1 !!!

        yt_map_idx_grid_flatten= get_yt_map_idx_grid(img_size, scale_factor).to(device) #shape: (img_size, img_size)

        depth= torch.tile(torch.arange(T).to(device).reshape(1, -1), (img_size*img_size, 1)).T.reshape(-1)
        column= torch.tile(torch.arange(img_size*img_size).to(device), (T,)) 
        row= torch.tile(yt_map_idx_grid_flatten[torch.arange(img_size*img_size).to(device)], (T,))

        A= A.index_put(indices=[depth, row, column], values=Ht_flatten.float()).unsqueeze(dim=0) ## CAUTION: MEMORY HUNGRY- 3 !!!
        return A
    
def convert_Ht2Atranspose(Ht, lambda_scale_factor):
    """
        Ht.shape: [1, T, img_size, img_size]
    """

    T, recon_img_size= Ht.shape[1], Ht.shape[2]
    scale_factor= 2**(lambda_scale_factor-1)
    yt_img_size= recon_img_size//scale_factor

    
    A= convert_Ht2A(Ht, lambda_scale_factor) # shape: (1, T, yt_img_size^2, recon_img_size^2)
    A_transpose_special= A.reshape(1, T*yt_img_size**2, recon_img_size**2).permute(0, 2, 1) # shape: (1, recon_img_size^2, T*yt_img_size^2)
    A_transpose_special_tiled = torch.tile(A_transpose_special.unsqueeze(dim=1), (1, T, 1, 1))# shape: (1, T, recon_img_size^2, T*yt_img_size^2)
    
    return A_transpose_special_tiled
    
def convert_Ht2Weights(Ht, scale_factor):
    """
    Ht.shape: [1, T, img_size, img_size]
    
    Returns:
    W.shape: [yt_img_size*yt_img_size, T, T*scale_factor*scale_factor]
    """
    
    T, recon_img_size= Ht.shape[1], Ht.shape[2]
    yt_img_size= recon_img_size//scale_factor
    
    
    Ht= Ht[0].permute(1,2,0) # shape: [img_size, img_size, T]
    W= Ht.reshape(yt_img_size,scale_factor,yt_img_size,scale_factor,T).permute(0, 2, 1, 3, 4) # shape: [yt_img_size, yt_img_size, scale_factor, scale_factor, T]
    W= W.flatten(start_dim= 0, end_dim=1).flatten(start_dim= 1, end_dim=2).permute(0,2,1)  # shape: [yt_img_size*yt_img_size, T, scale_factor*scale_factor]
    W= W.unsqueeze(dim= 2)  # shape: [yt_img_size*yt_img_size, T, 1, scale_factor*scale_factor]
    W= torch.tile(W, (1,1,T,1)).flatten(start_dim= 2, end_dim= 3)  # shape: [yt_img_size*yt_img_size, T, T*scale_factor*scale_factor]
    
    return W
    
    
    
    
    