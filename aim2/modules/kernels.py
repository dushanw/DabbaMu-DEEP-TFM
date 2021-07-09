
from numpy import pi, exp, sqrt
import numpy as np

def get_gaussian(side_len=5, s=1):
    k= (side_len-1)//2
    probs = [exp(-z*z/(2*s*s))/sqrt(2*pi*s*s) for z in range(-k,k+1)] 
    kernel = np.outer(probs, probs)
    return kernel


def impulse(side_len=5):
    mid = side_len//2
    kernel = np.zeros((side_len,side_len), dtype= 'float')
    kernel[mid, mid] = 1.0
    return kernel