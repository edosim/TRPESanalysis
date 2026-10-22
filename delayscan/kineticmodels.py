import numpy as np
from scipy.special import erf


def monoexp(t,k,sigma):
    """Monoexponential decay convolved with a Gaussian.

    Parameters
    ----------
    t : numpy.ndarray
        Time axis.
    k : float
        Rate constant.
    sigma : float
        Standard deviation of Gaussian.
    
    Returns
    -------
    numpy.ndarray
        Monoexpoential decay evaluated at ``t``.
    """
    return 0.5*np.exp(-k*t + 0.5*(k**2)*(sigma**2)) * (1 + erf((t-k*sigma**2)/(sigma*np.sqrt(2))))


class EAS_model():
    """Model using evolution-associated spectra (EAS) to describe time-resolved photoelectron
    spectra. 
    
    This class employs the sequential kinetic model
    A  -> B -> C, where A and B are two spectroscopically measurable states and C
    is a final non-measurable state outside the probing window.
    The photoelectron signal can thus be described as the sum of the contribution
    of A and B, S_A(eKE,t) + S_B(eKE,t), convolved with the instrument response 
    function (IRF), R(t) (assumed Gaussian)::

        S_A = c_A * e^(-k_1 * t) × R(t)
        S_B = c_B * k_1/(k_2 - k_1) * ((e^(-k_1 * t) - e^(-k_2 * t)) × R(t)

    where × denotes covolution and k_1 = 1/tau_1 and k_2 = 1/tau_2, and tau_1 and tau_2 are the
    time constants of the decay of A into B and B into C, respectively.

    Parameters
    ----------
    t0 : float
        Default time zero (maximum overlap of pump and probe pulse).
    sigma : float
        Standard deviation of the IRF.
    delays_length : int
        Default number of delays.
    energy_length : int
        Default number of energy values.
    """
    
    def __init__(self,t0, sigma, delays_length, energy_length):
        #time-zero
        self.t0 = t0
        # standard deviation =  cross-correlation / 2.355
        self.sigma = sigma
        #Number of delays
        self.delays_length = delays_length
        # Number of energy values
        self.energy_length = energy_length


    @staticmethod
    def B_model(t, A2, tau1, tau2, sigma):
        """Signal of species B as a function of time, convolved with the IRF.

        B is populated from A with time constant ``tau1`` and decays to C
        with time constant ``tau2``.
        
        Parameters
        ----------
        t : numpy.ndarray
            Delays relative to 0.
        A2 : float or numpy.ndarray
            Amplitude of B. Can be a scalar or an array of shape (n_energies,1).
        tau1 : float
            Time constant of formation of B from A.
        tau2 : float
            Time constant of decay of B to C.
        sigma : float
            Standard deviation of the IRF.
        
        Returns
        -------
        numpy.ndarray
            Convolved signal of B with the shape (n_energies, n_delays).

        """
        k1, k2 = 1/tau1, 1/tau2
        return A2*(k1/(k2-k1))*(monoexp(t,k1,sigma) - monoexp(t,k2,sigma))

    @staticmethod
    def A_model(t,A1,tau1,sigma):
        """Signal of species A as a function of time, convolved with the IRF.

        A is populated instantly at time zero and decays to B with a time constant of ``tau1``.
        
        Parameters
        ----------
        t : numpy.ndarray
            Delays relative to 0.
        A1 : float or numpy.ndarray
            Amplitude of A. Can be a scalar or an array of shape (n_energies,1).
        tau1 : float
            Time constant of decay of A to B.
        sigma : float
            Standard deviation of the IRF.
        
        Returns
        -------
        numpy.ndarray
            Convolved signal of A with the shape (n_energies, n_delays).

        """
        k1 = 1/tau1
        return A1*monoexp(t,k1,sigma)

    def kinetic_model(self,x, A1, A2, tau1, tau2,
                          t0 = None, sigma = None):
        """Total signal of species A and B as a function of time, convolved with the IRF.

        Parameters
        ----------
        x : numpy.ndarray
            Delays (not corrected for time-zero).
        A1 : float or numpy.ndarray
            Amplitude of A. Can be a scalar or an array of shape (n_energies,1).
        A2 : float or numpy.ndarray
            Amplitude of B. Can be a scalar or an array of shape (n_energies,1).
        tau1 : float
            Time constant of decay of A to B.
        tau2 : float
            Time constant of decay of B to C.
        t0 : float, optional
            Time zero. If None (default), uses ``self.t0``.
        sigma : float, optional
            Standard deviation of the IRF. If None (default), uses ``self.sigma``.
        
        Returns
        -------
        numpy.ndarray
            Convolved signal of A and B with the shape (n_energies, n_delays).

        """

        if t0 is None:
            t0 = self.t0
        if sigma is None:
            sigma = self.sigma

        return self.A_model(x-t0,A1,tau1,sigma) + self.B_model(x-t0,A2,tau1,tau2,sigma)



    def global_fit(self,flat_delays, tau1, tau2, *amplitudes,
                   energy_length = None,delays_length = None):
        """Model function to globally fit the whole delay-energy dataset.
        
        Suitable for ``scipy.optimize.curve_fit``. The time constants ``tau1`` and ``tau2``
        are shared for all energy channels, while each channel has its own amplitude for 
        A and B, i.e., the EAS.
        
        Parameters
        -----------
        flat_delays: numpy.ndarray
            Array of repeated delays with length ``(delays_length * energy_length)``. Reshaped 
            internally to ``(energy_length, delays_length)``.
        tau1 : float
            Time constant of decay of A to B.
        tau2 : float
            Time constant of decay of B to C.
        *amplitudes : float
            ``2 * energy_length`` amplitudes, ordered as ``(A1, A2)`` pairs for each energy
            channel. Reshaped internally to ``(energy_length, 2)``.
        energy_length : int, optional
            Number of energy values. If None (default), uses ``self.energy_length.``
        delays_length : int, optional
           Number of delays. If None (default), uses ``self.delays_length``.

        Returns
        -------
        numpy.ndarray
            Flattened modelled signal with length ``(delays_length * energy_length)``.
        """
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
    """Model using decay-associated spectra (DAS) to describe time-resolved photoelectron
    spectra. 
    
    This class models the signal as two exponential decays with time constants ``tau_1``
    and ``tau_2`` and amplitues ``A1`` and ``A2``, respectively, convolved with the instrument
    response function (IRF), R(t) (assumed Gaussian).

    Parameters
    ----------
    t0 : float
        Default time zero (maximum overlap of pump and probe pulse).
    sigma : float
        Standard deviation of the IRF.
    delays_length : int
        Default number of delays.
    energy_length : int
        Default number of energy values.
    """

    def __init__(self, t0, sigma, delays_length, energy_length):
        #time-zero
        self.t0 = t0
        # sigma = cross-correlation / 2.355
        self.sigma = sigma
        #Number of delays
        self.delays_length = delays_length
        # Number of energy values
        self.energy_length = energy_length


    def kinetic_model(self, x, A1, A2, tau1, tau2,
                            t0 = None, sigma = None):
        """Total signal of the two species as a function of time, convolved with the IRF.

        Parameters
        ----------
        x : numpy.ndarray
            Delays (not corrected for time-zero).
        A1 : float or numpy.ndarray
            Amplitude of first exponential. Can be a scalar or an array of shape (n_energies,1).
        A2 : float or numpy.ndarray
            Amplitude of second exponential. Can be a scalar or an array of shape (n_energies,1).
        tau1 : float
            Time constant of first exponential.
        tau2 : float
            Time constant of second exponential.
        t0 : float, optional
            Time zero. If None (default), uses ``self.t0``.
        sigma : float, optional
            Standard deviation of the IRF. If None (default), uses ``self.sigma``.
        
        Returns
        -------
        numpy.ndarray
            Convolved signal of the two exponenntials with the shape (n_energies, n_delays).
        """
        if t0 is None:
            t0 = self.t0
        if sigma is None:
            sigma = self.sigma

        k1, k2 = 1/tau1, 1/tau2

        return (A1 * monoexp(x-t0,k1,sigma) +
                A2 * monoexp(x-t0,k2,sigma))


    def global_fit(self, flat_delays, tau1, tau2, *amplitudes,
                   energy_length = None,delays_length = None):
        """Model function to globally fit the whole delay-energy dataset.
        
        Suitable for ``scipy.optimize.curve_fit``. The time constants ``tau1`` and ``tau2``
        are shared for all energy channels, while each channel has its own amplitude for 
        the first and second exponentials, i.e., the DAS.
        
        Parameters
        -----------
        flat_delays: numpy.ndarray
            Array of repeated delays with length ``(delays_length * energy_length)``. Reshaped 
            internally to ``(energy_length, delays_length)``.
        tau1 : float
            Time constant of first exponential.
        tau2 : float
            Time constant of second exponential.
        *amplitudes : float
            ``2 * energy_length`` amplitudes, ordered as ``(A1, A2)`` pairs for each energy
            channel. Reshaped internally to ``(energy_length, 2)``.
        delays_length : int, optional
           Number of delays. If None (default), uses ``self.delays_length``.
        energy_length : int, optional
            Number of energy values. If None (default), uses ``self.energy_length.``

        Returns
        -------
        numpy.ndarray
            Flattened modelled signal with length ``(delays_length * energy_length)``.

        """
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
    """ Model for retrieving evolution-associated spectra (EAS) from distorted time-resolved
    ultraviolet photoelectron spectra.

    This model combines the kinetic model of :class:`EAS_model` with the retrieval procedure of
    :class:`~delayscan.retrieve.JetFitFunc`. It employs the sequential kinetic model A -> B -> C,
    where A and B are two spectroscopically measurable states and C is a final non-measurable
    state outside the probing window. The true (undistorted) EAS of A and B are described by
    single Gaussians in eKE ``G_A(eKE)`` and ``G_B(eKE)``, and are converted into the distorted
    Gaussians ``g_A(eKE)`` and ``g_B(eKE)`` by a matrix ``basis``, which maps initial eKE to final
    eKE.

    The total photoelectron signal can be described as sum of the contributions of A and B,
    convolved with the instrument response function (IRF), R(t) (assumed Gaussian)::

        S_A(eKE, t) = g_A(eKE) * e^(-k_1 * t) × R(t)
        S_B(eKE, t) = g_B(eKE) * k_1/(k_2 - k_1) * ((e^(-k_1 * t) - e^(-k_2 * t)) × R(t)

    where × denotes covolution and k_1 = 1/tau_1 and k_2 = 1/tau_2, and tau_1 and tau_2 are the
    time constants of the decay of A into B and B into C, respectively.

    Parameters
    ----------
    t0 : float
        Default time zero (maximum overlap of pump and probe pulse).
    sigma : float
        Standard deviation of the IRF.
    delays : numpy.ndarray
        Pump-probe delays with shape (n_delays,). The number of delays is stored as ``delays_length``
    energy_length : int
        Default number of energy values.
    basis : numpy.ndarray
        (N, N) array containing the transformation between initial and final eKE.
    min_eKE : float
        Minimum eKE of the experimental spectrum. Usually spectra are cutoff at an eKE > 0 eV, but basis
        sets model down to 0 eV. Therefore, the model output is truncated to ``x >= min_eKE``.


    """

    A_model = staticmethod(EAS_model.A_model)
    B_model = staticmethod(EAS_model.B_model)
    
    def __init__(self,
                 t0,
                 sigma,
                 delays,
                 energy_length,
                 basis,
                 min_eKE):
        #time-zero
        self.t0 = t0
        # sigma = cross-correlation / 2.355
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
        """Evaluate sum of two distorted Gaussians at `x`.

        Each true spectrum is a Gaussian (``G_i``)::

            G_i =  amp_i * exp(-((x - ctr_i) / wid_i) ** 2)

        where ``wid_i = 0.60056 * fwhm_i``. 0.60056 is ``1 / (2 * sqrt(ln 2))``. 

        Each ``G_i`` is mapped to a final eKE distribution (``g_i``) using ``basis```::

            g_i = basis.T @ G_i
        
        The result is the sum of the two ``g_i`` truncated below ``min_eKE``.
            
        
        Parameters
        ----------
        x : numpy.ndarray
            eKE values of of shape (N,) or (N, n_delays) at which to evaluate the function.
            The first axis must match the dimensions of ``basis``.
        amp1 : float
            Height of Gaussian 1.
        ctr1 : float
            Centre of Gaussian 1.
        fwhm1 : float
            Full width at half maximum of Gaussian 1.
        amp2, ctr2, fwhm2 : float
            Same as ``amp1``, ``ctr1`` and ``fwhm1`` for Gaussian 2.
            
        Returns
        -------
        numpy.ndarray
            Sum of the two distorted Gaussians at points where ``x >= min_eKE``.
        """
        # fwhm to wid
        wid1,wid2 = fwhm1*0.60056,fwhm2*0.60056

        y1 =  amp1 * np.exp(-((x - ctr1)/wid1)**2)
        y2 =  amp2 * np.exp(-((x - ctr2)/wid2)**2)
        
        y1 = self.basis.T @ y1
        y2 = self.basis.T @ y2 
        return (y1 + y2)[x>=self.min_eKE]

    
    def global_fit(self,
                   flat_energies,amp1,ctr1,fwhm1,amp2,ctr2,fwhm2, tau1, tau2,
                   energy_length = None,delays_length = None):
        """Model function to globally fit the whole delay-energy dataset.
        
        The kinetic traces of A and B scale the amplitudes of two true Gaussians (``G_A`` and ``G_B``)
        at each delay. The true Gaussians are then distorted with ``basis`` and the sum of the distorted
        Gaussians (``g_A`` and ``g_B``) as a function of time are returned.

        Parameters
        ----------
        flat_energies : numpy.ndarray
            Array of repeated energies with length ``(delays_length * energy_length)``, e.g.
            ``np.tile(energies, delays_length)``. Reshaped internally to 
            ``(delays_length, energy_length)`` and transposed.
        amp1 : float
            Amplitude of the EAS of A.
        ctr1 : float
            Centre of the EAS of A.
        fwhm1 : float
            Full width at half maximum of the EAS of A.
        amp2, ctr2, fwhm2 : float
            Same as ``amp1``, ``ctr1`` and ``fwhm1`` for the EAS of B.
        tau1 : float
            Time constant of decay of A to B.
        tau2 : float
            Time constant of decay of B to C.
        energy_length : int, optional
            Number of eKE values. If None (default), uses ``self.energy_length``.
        delays_length : int, optional
            Number of delays. If None (default), uses ``self.delays_length``.

        Returns
        -------
        numpy.ndarray
            Flattened modelled (distorted) signal, at energies where ``eKE >= min_eKE``.
            Ordered with energy as the outer axis and delay as the inner axis,
            so the experimental data must have shape  ``(energy_length, delays_length)``).
        """
        if energy_length is None:
            energy_length = self.energy_length
        if delays_length is None:
            delays_length = self.delays_length

        energies = flat_energies.reshape(delays_length,energy_length).T

        # scale amps by trace
        a1 = self.A_model(self.delays,amp1,tau1,self.sigma)
        a2 = self.B_model(self.delays,amp2,tau1,tau2,self.sigma)

        return self.retrieval_function(energies,a1,ctr1,fwhm1,a2,ctr2,fwhm2).ravel()

