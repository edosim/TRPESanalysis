import numpy as np
from scipy.special import erf


def _monoexp(t,k,sigma):
    """Monoexponential helper function"""

    return 0.5*np.exp(-k*t + 0.5*(k**2)*(sigma**2)) * (1 + erf((t-k*sigma**2)/(sigma*np.sqrt(2))))


class EAS_model():
    """Model using evolution-associated spectra.
    It employes a sequential  kinetic model A  -> B -> C, where A and B are
    two spectroscopically measurable states and C is a final non-measurable state
    outside the probing window.
    The photoelectron signal can thus be described as the sum of the contribution
    of A and B, S_A(eKE,t) + S_B(eKE,t), convoluted with the instrument response function R(t).
    """
    
    def __init__(self,t0,sigma,delays_length,energy_length):
        #time-zero
        self.t0 = t0
        # sigma = cross-correlation / 2.335
        self.sigma = sigma
        #Number of delays
        self.delays_length = delays_length
        # Number of energy values
        self.energy_length = energy_length


    @staticmethod
    def B_model(t,A2,tau1,tau2,sigma):
        """Concentration of [B] as a function of time. Convoluted with Gaussian"""
        k1, k2 = 1/tau1, 1/tau2
        return A2*(k1/(k2-k1))*(_monoexp(t,k1,sigma) - _monoexp(t,k2,sigma))

    @staticmethod
    def A_model(t,A1,tau1,sigma):
        k1 = 1/tau1
        return A1*_monoexp(t,k1,sigma)

    def kinetic_model(self,x,A1,A2,tau1,tau2,
                          t0 = None,sigma = None):
        if t0 == None:
            t0 = self.t0
        if sigma == None:
            sigma = self.sigma

        return self.A_model(x-t0,A1,tau1,sigma) + self.B_model(x-t0,A2,tau1,tau2,sigma)



    def global_fit(self,flat_delays, tau1, tau2, *amplitudes):


        delays = flat_delays.reshape(self.energy_length, self.delays_length)
        amps = np.array(amplitudes).reshape(self.energy_length, 2)  

        return self.kinetic_model(
            delays,
            amps[:, 0][:, None], 
            amps[:, 1][:, None],
            tau1, tau2,
        ).ravel()



####################################

class DAS_model():
    """Model using decay-associated spectra.
       It employes the sum of two exponential decays,
       convoluted with the instrument response function.
    """
    
    def __init__(self, t0, sigma, delays_length, energy_length):
        #time-zero
        self.t0 = t0
        # sigma = cross-correlation / 2.335
        self.sigma = sigma
        #Number of delays
        self.delays_length = delays_length
        # Number of energy values
        self.energy_length = energy_length


    def kinetic_model(self, x, A1, A2, tau1, tau2,
                            t0 = None, sigma = None):
        if t0 == None:
            t0 = self.t0
        if sigma == None:
            sigma = self.sigma

        k1, k2 = 1/tau1, 1/tau2

        return (A1 * _monoexp(x-t0,k1,sigma) +
                A2 * _monoexp(x-t0,k2,sigma))


    def global_fit(self, flat_delays, tau1, tau2, *amplitudes):

        delays = flat_delays.reshape(self.energy_length, self.delays_length)
        amps = np.array(amplitudes).reshape(self.energy_length, 2)  

        return self.kinetic_model(
            delays,
            amps[:, 0][:, None], 
            amps[:, 1][:, None],
            tau1, tau2,
        ).ravel()