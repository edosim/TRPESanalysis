import numpy as np

def instrument_function(x,N0,l):
    return N0*np.exp((-x)/l)+1
