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
        ``amp_1, ctr_1, fwhm_1, amp_2, ctr_2, fwhm_2...``, where
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
    def __init__(self, num_surf, num_bulk, surf_bas, bulk_bas, min_eKE):
        self.num_surf = num_surf
        self.num_bulk = num_bulk
        self.num_peaks = self.num_surf + self.num_bulk
        self.surf_bas = surf_bas
        self.bulk_bas = bulk_bas
        self.min_eKE = min_eKE

    def f(self, x, *params):
        peaks_so_far = 0
        y = np.zeros_like(x)
 
        for i in range(0, len(params), 3):
            if peaks_so_far < self.num_surf:
                basis = self.surf_bas
            else:
                basis = self.bulk_bas
                
            amp = params[i]
            ctr = params[i+1]
            wid = params[i+2]*.60056
            y_tmp = amp * np.exp( -((x - ctr)/wid)**2)
            y += np.sum(basis.T * y_tmp, axis=1)
            
            peaks_so_far += 1
        
        x_above_cutoff = x[x>=self.min_eKE]
        y_above_cutoff = y[-x_above_cutoff.shape[0]:]
        
        return y_above_cutoff



def rebinData(data,min_eKE = 0, x_bins = None,nanWarningSuppress = False):
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

