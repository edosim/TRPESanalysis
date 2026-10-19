import numpy as np
from scipy.special import erf





def I(t,k,sigma):
    """Monoexponential helper function"""

    return 0.5*np.exp(-k*t + 0.5*(k**2)*(sigma**2)) * (1 + erf((t-k*sigma**2)/(sigma*np.sqrt(2))))

def B_model(t,A2,tau1,tau2,sigma):
    """Concentration of [B] as a function of time. Convoluted with Gaussian"""
    k1, k2 = 1/tau1, 1/tau2
    return A2*(k1/(k2-k1))*(I(t,k1,sigma) - I(t,k2,sigma))

def A_model(t,A1,tau1,sigma):
    return A1*I(t,1/tau1,sigma)
    

def kinetic_model_gau(x,A1,A2,tau1,tau2):
    sigma = 0.320 / 2.355  # cross-correlation with 250 & 267
    t0 = 0
    return A_model(x-t0,A1,tau1,sigma) + B_model(x-t0,A2,tau1,tau2,sigma)

def global_fit(flat_delays, tau1, tau2, *amplitudes):
    traces_number = 70
    points_per_trace = 47

    delays = flat_delays.reshape(traces_number, points_per_trace)
    amps = np.array(amplitudes).reshape(traces_number, 2)  

    return kinetic_model_gau(
        delays,
        amps[:, 0][:, None], 
        amps[:, 1][:, None],
        tau1, tau2,
    ).ravel()


class EAS_model():
    def __init__(self,t0,sigma):
        #time-zero
        self.t0 = None
        # sigma = cross-correlation / 2.335
        self.sigma = None
        # Number of energy values
        self.traces_number = 70
        #Number of delays
        self.points_per_trace = 47


    @staticmethod
    def _monoexp(t,k,sigma):
        """Monoexponential helper function"""

        return 0.5*np.exp(-k*t + 0.5*(k**2)*(sigma**2)) * (1 + erf((t-k*sigma**2)/(sigma*np.sqrt(2))))

    @staticmethod
    def B_model(t,A2,tau1,tau2,sigma):
        """Concentration of [B] as a function of time. Convoluted with Gaussian"""
        k1, k2 = 1/tau1, 1/tau2
        return A2*(k1/(k2-k1))*(I(t,k1,sigma) - I(t,k2,sigma))

    @staticmethod
    def A_model(t,A1,tau1,sigma):
        k1 = 1/tau1
        return A1*I(t,k1,sigma)

    def kinetic_model_gau(self,x,A1,A2,tau1,tau2,
                          t0 = None,sigma = None):
        if t0 == None:
            t0 = self.t0
        if sigma == None:
            sigma = self.sigma

        return A_model(x-t0,A1,tau1,sigma) + B_model(x-t0,A2,tau1,tau2,sigma)



    def global_fit(self,flat_delays, tau1, tau2, *amplitudes):


        delays = flat_delays.reshape(self.traces_number, self.points_per_trace)
        amps = np.array(amplitudes).reshape(self.traces_number, 2)  

        return kinetic_model_gau(
            delays,
            amps[:, 0][:, None], 
            amps[:, 1][:, None],
            tau1, tau2,
        ).ravel()
