import numpy as np
from os.path import getmtime
from pathlib import Path

import scipy

from .conversions import (
    tof_to_eke,
    pos_to_time,
    delaystrings_to_float,
    instrument_function
)



class DelayScan:
    """Time-resolved photoelectron spectroscopy data of a single delay scan.
     
    A delay scan is a set of time-of-flight (TOF) spectra recorded at different 
    pump-probe delay stage positions. At each delay pump-probe (pp), pump-only,
    probe-only and background (nothing) spectra are recorded. This class loads 
    the spectra, converts their TOF to electron kinetic energy (eKE), applies standard
    corrections and extracts the pump-probe signal free from pump-only, probe-only 
    and background contributions.
    
    Typical workflow
    -----------------
    >>> scan = DelayScan(**metadata)
    >>> scan.importData()
    >>> scan.applyJacobian()
    >>> scan.convert_to_eKE()
    >>> scan.applyInstrumentFunction()
    >>> scan.correctStreamingPotential()
    >>> scan.normaliseByIntegrationTime()
    >>> scan.subtract_background(ppInd,pumpInd,probeInd,nothingInd)
    >>> scan.trim_energies()

    Parameters
    ----------
    resultsFolder : str
        Folder containing the results. See Notes for the folder layout.
    date : str
        Date of measurements used in folder names.
    fileIndex : str
        Index of scan used in folder names.
    calParams : sequence of float
        TOF to eKE calibration parameters: flight length (m),
        time offset (ns), energy offset (eV). Passed to ``tof_to_eke``.
        The time offset (calParams[1]) is also used in the Jacobian.
    IFparams : sequence of float
        Parameters of instrument function: pre-exponential factor,
        decay constant. Passed to ``instrument_function``.
    IFcutoff : float
        Lowest eKE at which the instrument function correction gives
        reliable spectra. Used as default low-energy cutoff in ``trim_energies``.
    pumpWavelength : float
        Pump wavelength.
    probeWavelength : float
        Probe wavelength.
    pumpPulseEnergy : float
        Pump pulse energy.
    probePulseEnergy : float
        Probe pulse energy.
    integrationTime: float
        Acquisition time at each delay stage position.
        Used to normalise intensities.
    timeZero: float
        Approximate delay stage position of maximum overlap of pump and probe.
    SPbefore: sequence of float
        Streaming potential measurement taken before scan:
        streaming potential (eV), vacuum level offset (eV).
    SPafter: sequence of float
        Streaming potential measurement taken after scan:
        streaming potential (eV), vacuum level offset (eV).

    Attributes
    -------------
    streamingPotentialCorrection : float
        Streaming potential offset applied to eKE axis.
        Average of the sum of ``SPbefore`` and ``SPafter``.
    dataPath : str
        Path to folder containing data files.
    fileNames: list of str
        Names of data files in the order they are loaded.
    delays: numpy.ndarray
        Delay stage position (mm) one per set of pp, pump, 
        probe, nothing spectra.
    delayTimes : numpy.ndarray
        ``delays`` converted to time (ps) relative to ``timeZero``.
    TOFspectra : numpy.ndarray
        Raw spectra with shape ``(n_spectra, 2, n_TOFs). Axis 1 holds
        x values (ind 0) and y values (ind 1). 
    spectra : numpy.ndarray
        Copy of TOF spectra to which transformations and corrections
        are applied. 
    pps, pumps, probes, nothings : numpy.ndarray
        Spectra of each type per delay, with shape ``(n_delays, 2, n_energies)``.
    subtracted : numpy.ndarray
        Pump-probe spectra with pump-only, probe-only and background
        contributions subtracted, with shape ``(n_delays, 2, n_energies)``.
    subtracted_cut: numpy.ndarray
        ``subtracted`` within energy limit set by ``trim_energies``
    jacobianApplied, eKEconversionApplied, instrumentFunctionApplied, \
    streamingPotentialCorrected : bool
        Flags to ensure transformations and corrections are applied once.

    Notes
    -----
    Data are expected to be organised as::

        <resultsFolder>/<date>/<date><fileIndex>/<date><fileIndex>/

    with one text file per spectrum in the innermost folder. The first
    underscore-separated field of each file name must be the delay stage
    position, e.g. ``<delay>_<other fields>``. Files are read in order of
    modification time.
"""
    def __init__(self, **kwargs):
        """Store metadata and create path to data. Pass parameters as
        key arguments (see class docstring for list).
        """
        # Get all data and metadata
        self.resultsFolder = kwargs['resultsFolder']
        self.date = kwargs['date']
        self.fileIndex = kwargs['fileIndex']
        self.calParams = kwargs['calParams'] 
        self.IFparams = kwargs['IFparams']
        self.IFcutoff = kwargs['IFcutoff']
        self.pumpWavelength = kwargs['pumpWavelength']
        self.probeWavelength = kwargs['probeWavelength']
        self.pumpPulseEnergy = kwargs['pumpPulseEnergy']
        self.probePulseEnergy = kwargs['probePulseEnergy']
        self.integrationTime = kwargs['integrationTime']
        self.timeZero = kwargs['timeZero']
        self.SPbefore = np.array(kwargs['SPbefore']) 
        self.SPafter = np.array(kwargs['SPafter']) 
        self.streamingPotentialCorrection = sum(self.SPbefore+self.SPafter)/2
        
        # Path to where TR files are.
        self.dataPath = self.resultsFolder+rf"{self.date}/{self.date}{self.fileIndex}/{self.date}{self.fileIndex}/"

    def importData(self):
        """Imports TOF data from files and get delay values from file names.
        
        Files are loaded in order of modification time. The delay stage position is
        obtained from each file name as the first underscore-separated part.
        As each delay has four spectra (pp,pump,probe,nothing), delays are taken every 
        fourth entry.

        Sets ``fileNames``, ``delays``, ``delayTimes``, ``TOFspectra``, ``spectra``,
        and resets transformation flags to False.
        """
        # Get file names from folder
        self.directories = sorted(Path(self.dataPath).iterdir(), key=getmtime)
        self.fileNames = [str(d).split("\\")[-1] for d in self.directories]
        # Split by "_" to get delay stage position (1st item)
        self.file_names_split = [f.split('_') for f in self.fileNames]
        # Get delay strings
        self.delay_strings = np.array(self.file_names_split)[:,0]
        # Convert to float
        self.allDelays = delaystrings_to_float(self.delay_strings)
        # As there are four spectra for delay get list of unique delays
        self.delays = self.allDelays[0::4]
        # Convert to time
        self.delayTimes = pos_to_time(self.delays,self.timeZero)

        #Load data
        self.TOFspectra = []
        for fileName in self.fileNames:
            self.TOFspectra.append(np.loadtxt(self.dataPath+fileName).T)
        self.TOFspectra = np.array(self.TOFspectra)

        # Make copy so we can keep the original data and process it differently if needed
        self.spectra = np.copy(self.TOFspectra)

        #Checks to avoid transformations being applied twice
        self.jacobianApplied = False
        self.eKEconversionApplied = False
        self.instrumentFunctionApplied = False
        self.streamingPotentialCorrected = False
        self.integrationTimeNormalised = False
        

    
    def applyJacobian(self,scaling_factor = 1e-7):
        """Applies Jacobian transformation to the intensity of all spectra.

        Converting from a TOF to an energy x axis requires multiplying
        the intensities by (t - t0)^3 where t0 is the time offset in calParams.

        Does nothing if the Jacobian has already been applied.

        Parameters
        -----------
        scaling_factor : float, optional
            Arbitrary number to gives spectra intensities to around 0 and 1. Default = 1e-7."""
        if not self.jacobianApplied:            
            for i in range(len(self.spectra)):
                # Scales intensities by Jacobian
                self.spectra[i][1] *= ((self.TOFspectra[i][0]-self.calParams[1])**3)
                #Apply aribtrary scaling factor
                self.spectra[i][1] *= scaling_factor
            self.jacobianApplied = True


    def convert_to_eKE(self):
        """Applies TOF to eKE conversion to all spectra using ``calParams``.
        
        Does nothing if conversion has already been applied."""
        if not self.eKEconversionApplied:
            for i in range(len(self.spectra)):
                self.spectra[i][0] = tof_to_eke(self.spectra[i][0],*self.calParams)
            self.eKEconversionApplied = True


    def applyInstrumentFunction(self):
        """Applies instrument function correction to all spectra using ``IFparams``.

        Does nothing if correction has already been applied.
        """
        if not self.instrumentFunctionApplied:
            for i in range(len(self.spectra)):
                self.spectra[i][1] /= instrument_function(self.spectra[i][0],*self.IFparams)
            self.instrumentFunctionApplied = True

    def correctStreamingPotential(self):
        """Shifts energy axes by streaming potential and vacuum level offset using
        ``streamingPotentialCorrection``

        Does nothing if correction has already been applied.
        """
        if not self.streamingPotentialCorrected:
            for i in range(len(self.spectra)):
                self.spectra[i][0] += self.streamingPotentialCorrection
            self.streamingPotentialCorrected = True

    def normaliseByIntegrationTime(self):
        """Divide intensities by ``integrationTime``

        Does nothing if correction has already been applied.
        """
        if not self.integrationTimeNormalised:
            for i in range(len(self.spectra)):
                self.spectra[i][1] /= self.integrationTime
            self.integrationTimeNormalised = True


    def subtract_background(self,ppInd,pumpInd,probeInd,nothingInd):
        """Extract pp, pump, probe and nothing spectra
        and obtain background free signal by subtracting 
        pump, probe and nothing from pp.
        
        As spectra are recorded in groups of four the arguments give
        the indices of each of the four sets. These must be determined qualitatively
        by plotting the data (or quantitatively by maximum counts).
        
        Sets ``pps``, ``pumps``, ``probes``, ``nothings``
        and ``subtracted``, with shape ``(n_delays, 2, n_energies)``.
        
        Parameters
        ----------
        ppInd : int
            Index of pump-probe spectrum within each group of four.
        pumpInd : int
            Index of the pump-only spectrum.
        probeInd : int
            Index of the probe-only spectrum.
        nothingInd : int
            Index of the background (no pump, no probe) spectrum."""
        
        self.pps = self.spectra[ppInd::4]
        self.pumps = self.spectra[pumpInd::4]
        self.probes = self.spectra[probeInd::4]
        self.nothings = self.spectra[nothingInd::4]
        
        self.subtracted = np.copy(self.pps)
        self.subtracted[:,1]=self.subtracted[:,1]-self.pumps[:,1]-self.probes[:,1]-self.nothings[:,1]


    def trim_energies(self, low_eKE_cutoff = None, high_eKE_cutoff = None):
        """Removes data points below low eKE cutoff and above high eKE cutoff
        for ``subtracted``.
        
        Sets ``subtracted_cut``.

        Parameters
        ----------
        low_eKE_cutoff : float, optional
            Lower cutoff. Default is ``IFcutoff`` plus the
            streaming potential correction.
        high_eKE_cutoff : float, optional
            Upper cutoff. Defaults is 5.0 eV, the standard limit
            used in the lab.

        """
        if low_eKE_cutoff == None: 
            low_eKE_cutoff = self.IFcutoff+self.streamingPotentialCorrection
        if high_eKE_cutoff == None:
            high_eKE_cutoff = 5.0 # 5 eV is the standard limit we use in the lab

        mask = np.logical_and(self.subtracted[0,0,:]>=low_eKE_cutoff,
                              self.subtracted[0,0,:]<=high_eKE_cutoff)
        self.subtracted_cut = self.subtracted[:,:,mask]


def reduceBins(*args):
    """
    Call as:
        reduceBins(x, y, rebins)
        reduceBins(data, rebins)   # data is shape (2, N), rows are x and y
    Returns mean of x and y in data every `rebins` points.
    If rebins is not a factor of data.shape[1] values at the end of data are discarded

    Parameters
    ----------
    x : numpy.ndarray
        1D array of x values.
    y : numpy.ndarray
        1D array of y values, same length as ``x``.
    data : numpy.ndarray
        (2, N) array whose axis 0 holds x and y.
    rebins : int
        Number of bins to rebin the data into.

    Returns
    --------------
    numpy.ndarray
        Array of shape (2, N//rebins) with rebinned x (first row) and y (second row).
    
    
    Raises
    ----------
    TypeError
        If the number of arguments is not 2 or 3."""

    if len(args) == 3:
        x, y, rebins = args
    elif len(args) == 2:
        data, rebins = args
        x, y = data
    else:
        raise TypeError(
            f"reduceBins expects (x, y, rebins) or (data, rebins), got {len(args)} arguments"
        )

    if len(y) % rebins != 0:
        #Must trim end of x,y
        x=x[:-(len(x)%rebins)]
        y=y[:-(len(y)%rebins)]

    rebinned_x,rebinned_y = 0,0
    for i in range(rebins):
        rebinned_x += x[i::rebins]/rebins
        rebinned_y += y[i::rebins]/rebins
    return np.array((rebinned_x,rebinned_y))


class DelayScanSet(list):
    """ Set of time-resolved photoelectron spectroscopy scans from a single session.

    Each scan is a ``DelayScan`` object: a series of photoelectron spectra recorded at
    different pump-probe delays. Recording several scans under the same conditions and
    averaging them improves signal-to-noise ratio and allows better quality analysis.
    This class holds the individual scans, checks their compatibility and creates 
    the averaged data (with the option to rebin it). Scans must have been recorded
    under the same conditions (e.g. between the same two streaming potential measurements)
    and thus have the same energy and delay axes.

    The class is a ``list`` subclass so it supports its methods.
    
    Scans can be added before having been processed but they must be fully processed
    up to ``DelayScan.trim_energies()`` before running ``add_scans()``
    and ``rebin_subtracted_spectra(rebins)``.

    Typical workflow
    ----------------
    >>> scan_set = DelayScanSet(scan1, scan2, scan3)
    >>> scan_set.add_scans()
    >>> scan_set.rebin_subtracted_spectra(rebins)

    Parameters
    -----------
    *args: DelayScan
        The DelayScan objects to include.

    Attributes
    ----------
    subtracted : numpy.ndarray
        Average of background-subtracted pump-probe intensites normalised to its
        maximum, with shape ``(n_delays, n_energies)``.
    energies : numpy.ndarray
        eKE axis shared by all scans with length ``n_energies``
    delays : numpy.ndarray
        Delay times shared by all scans with length ``n_delays``
    subtracted_rebinned : numpy.ndarray
        ``subtracted`` after rebinning along the energy axis normalised to its 
        maximum, with shape ``(n_delays, n_energies_rebinned)``.
    energies_rebinned : numpy.ndarray
        ``energies``` after rebinning, with len ``n_energies_rebinned``.
    rebins : int
        Number of new bins to rebin data into.
    t0 : float
        t0 obtained by fitting.
    delays_t0 : numpy.ndarray
        ``delays`` shifted by t0.    
 
"""


    def __init__(self, *args):
        """Create set from scans

        Parameters
        -----------
        *args: DelayScan
            The DelayScan objects to include.
        """
        super().__init__(args)


    def energies_match(self):
        """Check whether all scans share the same energy axis.
        
        Compares the trimmed energies of ``subtracted_cut`` to the first
        scan in the set.
        
        Returns
        ----------
        bool
            True if all the energy axes match, False otherwise.
        """
        #Get trimmed energies of first scan
        energies_reference = self[0].subtracted_cut[0,0,:]
        #Compare it to the rest
        for scan in self[1:]:
            energies_current = scan.subtracted_cut[0,0,:]
            if not np.allclose(energies_current, energies_reference):
                return False
        return True

    def delays_match(self):
        """Check whether all scans share the same delays.
        
        Compares the ``delayTimes`` of each scan to the first
        scan in the set.
        
        Returns
        ----------
        bool
            True if all the delays match, False otherwise.
        """
        delays_reference = self[0].delayTimes
        for scan in self[1:]:
            delays_current = scan.delayTimes
            if not np.allclose(delays_current, delays_reference):
                return False
        return True

    def shapes_match(self):
        """Check whether the all scans have the same data shape.

        Compares the shape of ``subtracted_cut`` of each scan to the first
        scan in the set.

        Returns
        -------------
        bool
            True if all the data has the same shape, False otherwise.
        """
        shape_reference = self[0].subtracted_cut.shape
        for scan in self[1:]:
            if not (scan.subtracted_cut.shape == shape_reference):
                return False
        return True             


    def add_scans(self):
        """ Averages the trimmed subtracted spectra of all scans.

        First checks if energies, delays and data shapes are compatible. Then
        averages the intensities in ``subtracted_cut`` and normalises the result
        by its maximum value.

        Raises
        --------
        ValueError
            If energies, delays or data shapes in scans do not match.
        
        """
        #Check that the eKEs of all scans are the same
        if not self.energies_match():
            raise ValueError("The energy axes of single scans do not match")
        #Check that the delays of all scans are the same
        if not self.delays_match():
            raise ValueError("The delay axes of single scans do not match")
        #Check that they all have the same shape
        if not self.shapes_match():
            raise ValueError("The scans do not all have the same shape")

       # Create empty array with same shape as the individual subtracted data
        self.subtracted = np.zeros_like(self[0].subtracted_cut[:,1])
        #Sum over scans in scanSet scaled by the number of scans (i.e. average)
        for scan in self:
            self.subtracted += scan.subtracted_cut[:,1]/len(self)

        #Normalise by highest intensity
        self.subtracted /= np.max(self.subtracted)
        #Save rebinned_energies
        self.energies = scan.subtracted_cut[0,0]
        #Save delays
        self.delays = scan.delayTimes



    def rebin_subtracted_spectra(self,rebins):
        """ Rebin the averaged spectra along the energy axis.

        Each spectrum in ``subtracted`` is rebinned every ``rebins`` bins.
        The resulting data set is normalised to its maximum.

        Sets ``subtracted_rebinned``, ``energies_rebinned`` and ``rebins``.
        
        """
        subtracted_rebinned = []
        for spectrum in self.subtracted:
            rebinned_spectrum = reduceBins(self.energies,spectrum,rebins)[1]
            subtracted_rebinned.append(rebinned_spectrum)
        
        self.subtracted_rebinned = np.array(subtracted_rebinned)

        #Normalise by highest intensity
        self.subtracted_rebinned /= np.max(self.subtracted_rebinned)
        #Save rebinned_energies
        self.energies_rebinned = reduceBins(self.energies,spectrum,rebins)[0]
        #Save rebins
        self.rebins = rebins

    def set_time_zero(self,t0):
        """Sets time zero and shifts the delay axis.
        
        As ``DelayScan.timeZero`` is only an initial estimate of time zero, this method
        can be called to set a more accurate time zero and obtain the delays relative to it.
        
        Sets ``t0`` and ``delays_t0``
        
        Parameters
        -----------
        t0 : float
            New time zero value (ps)
        """
        self.t0 = t0
        self.delays_t0 = self.delays - t0




def create_save_arr(x, y, Z):
    """Create a 2D array with the x axis along the top row,
    the y axis down the first column, and Z in
    the body. The top-left corner is a placeholder zero.
    
    Parameters
    ----------
    x : array_like
        1D array of x axis values.
    y : array_like
        1D array of y axis values.
    Z : array_like
        Array of shape (len(x),len(y)) of z values.

    Returns
    ----------
    numpy.ndarray
        Array of shape (len(x)+1, len(y)+1).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    Z = np.asarray(Z)

    # Add y to the left
    save_arr = np.vstack((y, Z.T))
    # Add leading 0 to x
    x = np.hstack(([0,],x))
    # Add x to the top
    save_arr = np.vstack((x,save_arr.T))
    
    return save_arr


### Analysis functions
def monoexponential_witht0(x,A1,tau1,t0,sigma):
    """Return a monoexponential decay convoluted with a Gaussian.
    
    Parameters
    ----------
    x : numpy.ndarray
        Input x values.
    A1 : float or numpy.ndarray
        Amplitude of decay.
    tau1 : float or numpy.ndarray
        Decay constant.
    t0 : float
        Time zero (x axis offset).
    sigma : float
        Sigma of Gaussian.
    
    Returns
    ----------
    scalar or numpy.ndarray
        Values of function at ``x``.
"""
    t = x - t0
    return (A1/2 * np.exp(-1/tau1 * t + 0.5 * (sigma**2)/(tau1**2)) *
           (1 + (scipy.special.erf((t-(sigma**2)/tau1) / (np.sqrt(2.0)*sigma)))))


def monoexponential(x,A1,tau1,sigma):
    """Return a monoexponential decay convoluted with a Gaussian centred at 0.
    
    Parameters
    ----------
    x : numpy.ndarray
        Input x values.
    A1 : float or numpy.ndarray
        Amplitude of decay.
    tau1 : float or numpy.ndarray
        Decay constant.
    sigma : float
        Sigma of Gaussian.
    
    Returns
    ----------
    scalar or numpy.ndarray
        Values of function at ``x``.
"""
    t0 = 0
    return monoexponential_witht0(x,A1,tau1,t0,sigma)
