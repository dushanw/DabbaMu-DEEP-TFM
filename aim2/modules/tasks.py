import torch
from kornia.morphology import closing

def segment(images): #images.shape: (b, 1, 32, 32)
    thresh= (310- 134.28)/500.0 #0.33144 #0.53144
    
    kernel = torch.ones(10, 10).to(images.device)
    closed_images = closing((images > thresh).float(), kernel)
    return closed_images