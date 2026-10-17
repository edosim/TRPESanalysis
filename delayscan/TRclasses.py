import numpy as np
from os.path import getmtime
from pathlib import Path

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
        # As there are for spectra for delay get list of unique delays
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
        """Coorects for SP and vacuum level"""
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

def reduceBins(data, rebins):
    """Returns mean of x and y in data every `bins` points.
    If bins is not a factor of data.shape[1] values at the end of data are discarded"""
    x,y=data
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
    """This is just a list but I may add stuff to it in the future."""
    def __init__(self, *args):
        super().__init__(args)

