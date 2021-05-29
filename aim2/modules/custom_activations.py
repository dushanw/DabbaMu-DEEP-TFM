
import torch
def sigmoid_custom(x, m):
    return 1/(1+torch.exp(-m*x))

def sigmoid_custom2(x, m, c= 10):
    return 1/(1+torch.exp(-m*x+m*c))

def sigmoid_custom2_with_translation(x, m, c=10, p=1, q=0):
    x_dash= p*x+ q
    return 1/(1+torch.exp(-m*x_dash + m*c))

def unit_func(x, m):
    return x


def inc_m(m, epoch, n=None):
    if n==None:n=1
    m+=n
    return m