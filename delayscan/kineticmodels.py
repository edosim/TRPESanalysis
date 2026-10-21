import numpy as np
from scipy.special import erf


def monoexp(t,k,sigma):
    """Monoexponential helper function"""

    return 0.5*np.exp(-k*t + 0.5*(k**2)*(sigma**2)) * (1 + erf((t-k*sigma**2)/(sigma*np.sqrt(2))))


class EAS_model():
    """Model using evolution-associated spectra (EAS) to model time-resolved photoelectron
    spectra. 
    
    This class employs the sequential kinetic model
    A  -> B -> C, where A and B are two spectroscopically measurable states and C
    is a final non-measurable state outside the probing window.
    The photoelectron signal can thus be described as the sum of the contribution
    of A and B, S_A(eKE,t) + S_B(eKE,t), convoluted with the instrument response 
    function R(t) (assumed Gaussian).
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
        return A2*(k1/(k2-k1))*(monoexp(t,k1,sigma) - monoexp(t,k2,sigma))

    @staticmethod
    def A_model(t,A1,tau1,sigma):
        k1 = 1/tau1
        return A1*monoexp(t,k1,sigma)

    def kinetic_model(self,x,A1,A2,tau1,tau2,
                          t0 = None,sigma = None):
        if t0 is None:
            t0 = self.t0
        if sigma is None:
            sigma = self.sigma

        return self.A_model(x-t0,A1,tau1,sigma) + self.B_model(x-t0,A2,tau1,tau2,sigma)



    def global_fit(self,flat_delays, tau1, tau2, *amplitudes,
                   energy_length = None,delays_length = None):
        if energy_length is None:
            energy_length = self.energy_length
        if delays_length is None:
            delays_length = self.delays_length

        delays = flat_delays.reshape(energy_length, delays_length)
        amps = np.array(amplitudes).reshape(energy_length, 2)  

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
        if t0 is None:
            t0 = self.t0
        if sigma is None:
            sigma = self.sigma

        k1, k2 = 1/tau1, 1/tau2

        return (A1 * monoexp(x-t0,k1,sigma) +
                A2 * monoexp(x-t0,k2,sigma))


    def global_fit(self, flat_delays, tau1, tau2, *amplitudes,
                   energy_length = None,delays_length = None):
        if energy_length is None:
            energy_length = self.energy_length
        if delays_length is None:
            delays_length = self.delays_length

        delays = flat_delays.reshape(energy_length, delays_length)
        amps = np.array(amplitudes).reshape(energy_length, 2)  

        return self.kinetic_model(
            delays,
            amps[:, 0][:, None], 
            amps[:, 1][:, None],
            tau1, tau2,
        ).ravel()


######################

class RetrieveModel():
    """
    Fits two distorted gaussians to the data following the EAS model (A -> B -> C)
    """
    def __init__(self,
                 t0,
                 sigma,
                 delays,
                 energy_length,
                 basis,
                 min_eKE):
        #time-zero
        self.t0 = t0
        # sigma = cross-correlation / 2.335
        self.sigma = sigma
        # delays
        self.delays = delays
        #Number of delays
        self.delays_length = len(delays)
        # Number of energy values
        self.energy_length = energy_length

        #basis set
        self.basis = basis
        #min_eKE for cutting off result
        self.min_eKE = min_eKE


    def retrieval_function(self,x,amp1,ctr1,fwhm1,amp2,ctr2,fwhm2):
        # fwhm to wid
        wid1,wid2 = fwhm1*0.60056,fwhm2*0.60056

        y1 =  amp1 * np.exp(-((x - ctr1)/wid1)**2)
        y2 =  amp2 * np.exp(-((x - ctr2)/wid2)**2)
        
        y1 = self.basis.T @ y1
        y2 = self.basis.T @ y2 
        return (y1 + y2)[x>=self.min_eKE]


    @staticmethod
    def B_model(t,A2,tau1,tau2,sigma):
        """Concentration of [B] as a function of time. Convoluted with Gaussian"""
        k1, k2 = 1/tau1, 1/tau2
        return A2*(k1/(k2-k1))*(monoexp(t,k1,sigma) - monoexp(t,k2,sigma))

    @staticmethod
    def A_model(t,A1,tau1,sigma):
        k1 = 1/tau1
        return A1*monoexp(t,k1,sigma)


    
    def global_fit(self,
                   flat_energies,amp1,ctr1,fwhm1,amp2,ctr2,fwhm2, tau1, tau2,
                   energy_length = None,delays_length = None):

        if energy_length is None:
            energy_length = self.energy_length
        if delays_length is None:
            delays_length = self.delays_length

        energies = flat_energies.reshape(delays_length,energy_length).T

        # scale amps by trace
        a1 = self.A_model(self.delays,amp1,tau1,self.sigma)
        a2 = self.B_model(self.delays,amp2,tau1,tau2,self.sigma)

        return self.retrieval_function(energies,a1,ctr1,fwhm1,a2,ctr2,fwhm2).ravel()

