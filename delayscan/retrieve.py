import numpy as np


# Multiple Gaussian function for fitting to bases
def multiple_gaussians(x, *params):
    """
    Return sum of Gaussians evaluated at x of the form::

        y_i = amp_i * exp(-((x - ctr_i) / wid_i) ** 2)
    
    where ``wid_i = 0.60056 * fwhm_i``. 0.60056 is ``1 / (2 * sqrt(ln 2))``. 

    Parameters
    ----------
    x : numpy.ndarray
        Points at which to evaluate the function
    *params : float
        A series of three values per Gaussian in the order 
        ``amp_1, ctr_1, fwhm_1, amp_2, ctr_2, fwhm_2, ...``, where
        amp_i is the height, ctr_i is the centre and
        fwhm_i is the full width at half maximum.

    Returns
    -------
    numpy.ndarray
        Sum of all Gaussians at ``x``.
    """
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        amp = params[i]
        ctr = params[i+1]
        fwhm = params[i+2] 
        wid = fwhm * 0.60056
        y = y + amp * np.exp( -((x - ctr)/wid)**2)
    return y


class JetFitFunc():
    """Ultraviolet liquid-microjet photoelectron spectra are distorted by rovibrational
    inelastic scattering of photoelectrons inside the liquid. Monte-Carlo electron 
    scattering simulations can be employed to obtain a matrix mapping initial electron
    kinetic energy (eKE) to final eKE, which allows true (undistorted) photoelectron
    spectra to be retrieved from distorted experimental ones.
    
    This class stores the mapping matrix and provides the function ``f`` which performs
    the mapping and is suitable for ``scipy.optimize.curve_fit``.
    The true spectrum is modelled as a sum of Gaussians each of which is tranformed 
    using one of two basis sets, depending on where electrons originate:

    - ``'B'`` (bulk): electrons are generated uniformly inside the liquid jet (e.g. the solvent).
    These use ``bulk_basis``.
    - ``'S'`` (surface): electrons are generated a depth-dependent distribution (e.g. surfactants).
    These use ``surf_basis``.
    
    Parameters
    ----------
    gauss_types : sequence of {'B', 'S'}
        One entry per Gaussian, giving the basis set used for each set of three
        parameters (``amp``, ``ctr``, ``fwhm``) passed to ``f``. ``'S'`` selects
        ``surf_basis`` and ``'B'`` selects ``bulk_basis``.
    surf_basis : numpy.ndarray
        (N, N) array containing the transformation between initial and final eKE for
        surface-enhanced species.
    bulk_basis : numpy.ndarray
        (N, N) array containing the transformation between initial and final eKE for
        uniformly distributed species.
    min_eKE : float
        Minimum eKE of the experimental spectrum. Usually spectra are cutoff at an eKE > 0 eV, but basis
        sets model down to 0 eV. Therefore, the model output is truncated to ``x >= min_eKE``.

    Raises
    ------
    ValueError
        If ``gauss_types`` contains anything other than ``'S'`` or ``'B'``.

    Notes
    -----
    While `f` outputs ``y`` truncated above ``min_eKE``, the eKE grid ``x`` must have 
    N points, i.e., the same as one of the dimensions of the bases. Additionally ``x``
    should have the same values employed in the scattering simulation
    (usually ``np.arange(0.01,5.01,0.01)``).
"""
    def __init__(self, gauss_types, surf_basis, bulk_basis, min_eKE):
        """
        Store bases, types of Gaussian (bulk or surface) and minimum eKE.

        See the class docstring for the full list of parameters.
        """
        # Check  gauss_types only contains 'S' and 'B'
        invalid = [g for g in gauss_types if g not in ('S', 'B')]
        if invalid:
            raise ValueError(f"gauss_types should contain 'S' and 'B' only, got {invalid}.")
        self.gauss_types = gauss_types # list of 'S' (surface) and 'B' (bulk)
        self.surf_basis = surf_basis #basis set to use for 'S' gaussians
        self.bulk_basis = bulk_basis #basis set to use for 'B' gaussians
        self.min_eKE = min_eKE #minimum eKE to fit to

    def f(self, x, *params):
        """Evaluate sum of distorted Gaussians evaluated at `x`.

        Each true spectrum is a Gaussian (``G_i``)::

            G_i =  amp_i * exp(-((x - ctr_i) / wid_i) ** 2)

        where ``wid_i = 0.60056 * fwhm_i``. 0.60056 is ``1 / (2 * sqrt(ln 2))``. 

        Each ``G_i`` is mapped to a final eKE distribution (``g_i``) using the 
        basis set selected by ```gauss_types[i]```::

            g_i = basis.T @ G_i
        
        The result is the sum of all ``g_i`` truncated below ``min_eKE``.
            
        
        Parameters
        ----------
        x : numpy.ndarray
            eKE values of length N at which to evaluate the function.
            Must have the same dimension N of the basis sets.
        *params : float
            A series of three values per Gaussian in the order 
            ``amp_1, ctr_1, fwhm_1, amp_2, ctr_2, fwhm_2, ...``, where
            amp_i is the height, ctr_i is the centre and
            fwhm_i is the full width at half maximum of the true Gaussian (``G_i``)

        Returns
        -------
        numpy.ndarray
            Sum of all distorted Gaussians at points where ``x >= min_eKE``.
        """

        y = np.zeros_like(x)
 
        for gauss_ind, param_ind in enumerate(range(0, len(params), 3)):
            if self.gauss_types[gauss_ind] == "S":
                basis = self.surf_basis
            elif self.gauss_types[gauss_ind] == "B":
                basis = self.bulk_basis
                
            amp = params[param_ind]
            ctr = params[param_ind+1]
            wid = params[param_ind+2]*.60056

            y_tmp = amp * np.exp( -((x - ctr)/wid)**2)
            y += basis.T @ y_tmp
            
        y_above_cutoff = y[x>=self.min_eKE]
        
        return y_above_cutoff

def rebinData(data,min_eKE = 0, x_bins = None,nanWarningSuppress = False):
    """
    Rebin a spectrum onto a regular eKE grid by averaging within each bin.

    Each y value is assigned to the bin containing its x value, and the y values
    in the same bin are averaged. Bins at or below ``min_eKE`` are discarded, and
    any empty bins (which would be ``nan``) are set to 0.

    Parameters
    ----------
    data : numpy.ndarray
        (2, N) array, where ``data[0]`` holds the x values (eKE, eV)
        and ``data[1]`` holds the corresponding y values (intensity).
    min_eKE : float, optional
        Bins whose right edge is not greater than this value (eV) are removed
        from the output. Defaults to 0.
    x_bins : numpy.ndarray, optional
        Monotonically increasing bin edges (eV), giving ``len(x_bins) - 1`` bins.
        Each bin includes its left edge and excludes its right edge. Defaults to
        ``np.arange(0.00, 5.01, 0.01)``.
    nanWarningSuppress : bool, optional
        If False (default), print a message giving the lowest eKE at which an
        empty bin was found. If True, suppress the message.
    Returns
    -------
    numpy.ndarray
        (2, K) array, where ``M`` is the number of bins above ``min_eKE``.
        The first row contains the right bin edges and the second row contains
        the mean y value in each bin, with empty bins set to 0.

    Notes
    -----
    - x values outside the range of ``x_bins`` are ignored.
    - Empty bins give ``nan`` when averaged. These are expected only at high eKE
    where the difference between consecutive values gets larger, and are set to 0.
    - The right bins are selected because this is what is done in the Monte-Carlo
    simulation used to model the distortion of photoelectron spectra.
    
    """
    # Default bin edges if none selected
    if x_bins is None:
        x_bins = np.arange(0.00,5.01,0.01)

    # Get indices of the bins to which each value in x belongs
    bin_indices = np.digitize(data[0],x_bins)
    # !: This returns 1-based indices, i.e. the indices of the right edges
    #So you must subtract 1
    bin_indices -= 1
    
    # Bin y values according to bin_indices and average them
    binned_y = x_bins[1:]*0
    for i, bin in enumerate(binned_y):
        binned_y[i] = np.mean(data[1][bin_indices==i])

    # Remove data below min_eKE for retrieval fit
    binned_x = x_bins[1:][x_bins[1:]>min_eKE]
    binned_y = binned_y[x_bins[1:]>min_eKE]

    #If no value is present between 2 bins nan will be returned
    #However nan cannot be used in the fit
    # one way around is to interpolate thedata where nans are present
    # Otherwise we can set the data above the first nan to 0
    # As they appear at high eKEs only
    if any(np.isnan(binned_y)):
        #Find min x value with corresponidng y = nan
        min_nan_x = min(binned_x[np.isnan(binned_y)])
        if not nanWarningSuppress:
            print(f"'nan' values present above {min_nan_x:.2f}. y set ot 0 where y ='nan' above this value.")
        #Remove data above first nan
        binned_y[np.isnan(binned_y)] = 0
    
    return np.asarray((binned_x,binned_y))

