import torch

def ifft(x): #input: frequency domain ; output: spatial domain
    x= torch.abs(torch.fft.ifft(x))
    return x

def fft(x): #input: spatial domain ; output: frequency domain
    x= torch.abs(torch.fft.fft(x))
    return x