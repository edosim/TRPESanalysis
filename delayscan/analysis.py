import numpy as np
import scipy
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.special import erf

def monoexponential_witht0(x,A1,tau1,t0,sigma):
    
    return (A1 * np.exp((-1/tau1)*(x-t0))*np.exp((sigma**2)/(2*(tau1**2))) *
           (1 + (scipy.special.erf((x-t0-((sigma**2)/tau1))/(np.sqrt(2.0)*sigma)))))


