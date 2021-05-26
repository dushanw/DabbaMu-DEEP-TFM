
from numpy import pi, exp, sqrt
import numpy as np

def get_gaussian(side_len=5, s=1):
    k= (side_len-1)//2
    probs = [exp(-z*z/(2*s*s))/sqrt(2*pi*s*s) for z in range(-k,k+1)] 
    kernel = np.outer(probs, probs)
    return kernel