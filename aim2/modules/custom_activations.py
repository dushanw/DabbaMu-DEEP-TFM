
import torch
def sigmoid_custom(x, m):
    return 1/(1+torch.exp(-m*x))

def sigmoid_custom2(x, m, c= 10):
    return 1/(1+torch.exp(-m*x+m*c))


def inc_m(m, epoch, n=None):
    if n==None:n=1
    m+=n
    return m