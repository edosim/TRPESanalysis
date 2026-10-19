import numpy as np
from os.path import getmtime
from pathlib import Path

import scipy

from .conversions import (
    tof_to_eke,
    pos_to_time,
    delaystrings_to_float
)

from .calibrations import instrument_function


class DelayScan:
    def __init__(self, **kwargs):
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
        self.SPbefore = kwargs['SPbefore'] 
        self.SPafter = kwargs['SPafter'] 
        self.streamingPotentialCorrection = sum(self.SPbefore+self.SPafter)/2
        
        # Path to where TR files are.
        self.dataPath = self.resultsFolder+rf"{self.date}\{self.date}{self.fileIndex}\{self.date}{self.fileIndex}/"

    def importData(self):
        """Imports TOF data from file"""
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
        """Applies Jacobian transformation to all spectra.
        Scaling factor is an arbitrary number that gives spectra "nice" y values"""
        if not self.jacobianApplied:            
            for i in range(len(self.spectra)):
                # Scales intensities by Jacobian
                self.spectra[i][1] *= ((self.TOFspectra[i][0]-self.calParams[1])**3)
                #Apply aribtrary scaling factor
                self.spectra[i][1] *= scaling_factor
            self.jacobianApplied = True


    def convert_to_eKE(self):
        """Applies TOF to eKE conversion to all spectra"""
        if not self.eKEconversionApplied:
            for i in range(len(self.spectra)):
                self.spectra[i][0] = tof_to_eke(self.spectra[i][0],*self.calParams)
            self.eKEconversionApplied = True


    def applyInstrumentFunction(self):
        """Applies instrument function to all spectra"""
        if not self.instrumentFunctionApplied:
            for i in range(len(self.spectra)):
                self.spectra[i][1] /= instrument_function(self.spectra[i][0],*self.IFparams)
            self.instrumentFunctionApplied = True

    def correctStreamingPotential(self):
        """Corrects for streaming and vacuum level offset"""
        if not self.streamingPotentialCorrected:
            for i in range(len(self.spectra)):
                self.spectra[i][0] += self.streamingPotentialCorrection
            self.streamingPotentialCorrected = True

    def normaliseByIntegrationTime(self):
        """Normalise all spectra by integration time"""
        if not self.integrationTimeNormalised:
            for i in range(len(self.spectra)):
                self.spectra[i][1] /= self.integrationTime
            self.integrationTimeNormalised = True


    def subtract_background(self,ppInd,pumpInd,probeInd,nothingInd):
        """Extracts pp, pump, probe and nothing spectra
        and subtracts pump, probe and nothing from pp"""
        
        self.pps = self.spectra[ppInd::4]
        self.pumps = self.spectra[pumpInd::4]
        self.probes = self.spectra[probeInd::4]
        self.nothings = self.spectra[nothingInd::4]
        
        self.subtracted = np.copy(self.pps)
        self.subtracted[:,1]=self.subtracted[:,1]-self.pumps[:,1]-self.probes[:,1]-self.nothings[:,1]


    def trim_energies(self, low_eKE_cutoff = None, high_eKE_cutoff = None):
        """Removes data points below low eKE cutoff and above high eKE cutoff"""
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
    Returns mean of x and y in data every `bins` points.
    If bins is not a factor of data.shape[1] values at the end of data are discarded"""

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
    """This is just a list with the date(s) but I may add stuff to it in the future."""
    def __init__(self, *args):
        super().__init__(args)


    @property
    def date(self):
        date = {scan.date for scan in self}
        for scan in self:
            date.add(scan.date)

        return date


    def energies_match(self):
        energies_reference = self[0].subtracted_cut[0,0,:]
        for scan in self[1:]:
            energies_current = scan.subtracted_cut[0,0,:]
            if not np.allclose(energies_current, energies_reference):
                return False
        return True

    def delays_match(self):
        #Check delays are the same
        delays_reference = self[0].delayTimes
        for scan in self[1:]:
            delays_current = scan.delayTimes
            if not np.allclose(delays_current, delays_reference):
                return False
        return True

    def shapes_match(self):
        shape_reference = self[0].subtracted_cut.shape
        for scan in self[1:]:
            if not (scan.subtracted_cut.shape == shape_reference):
                return False
        return True             


    def add_scans(self):
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
        """"""
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





def create_save_arr(x, y, Z):
    """Outputs a 2D array with the x axis along the top row,
    the y axis down the first column, and Z in
    the body. The top-left corner is a placeholder zero."""
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
    
    return (A1 * np.exp((-1/tau1)*(x-t0))*np.exp((sigma**2)/(2*(tau1**2))) *
           (1 + (scipy.special.erf((x-t0-((sigma**2)/tau1))/(np.sqrt(2.0)*sigma)))))
