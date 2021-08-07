import torch

def ifft(x): #input: frequency domain ; output: spatial domain
    x= torch.abs(torch.fft.ifft(x))
    return x

def fft(x): #input: spatial domain ; output: frequency domain
    x= torch.abs(torch.fft.fft(x))
    return x

def fft_2d(x): #input: spatial domain ; output: frequency domain
    x= torch.abs(torch.fft.fft2(x))
    return x

def ifft_2d(x): #input: frequency domain ; output: spatial domain
    x= torch.abs(torch.fft.ifft2(x))
    return x

def fft_2d_with_fftshift(x): #input: spatial domain ; output: frequency domain
    x= torch.abs(torch.fft.fftshift(torch.fft.fft2(x), dim= (2,3)))
    return x
def ifft_2d_with_fftshift(x): #input: frequency domain ; output: spatial domain
    x= torch.abs(torch.fft.fftshift(torch.fft.ifft2(x), dim= (2,3)))
    return x

def ifft_2d_with_fftshift_real(x): #input: frequency domain ; output: spatial domain
    x_ifft2 = torch.fft.ifft2(x)
    #print(f'ifft_converted_weights : {x_ifft2[0,0,0]}')
    
    x= torch.fft.fftshift(x_ifft2, dim= (2,3))
    
    return x.real

def identity(x):
    return x

def absolute(x):
    return torch.abs(x)
