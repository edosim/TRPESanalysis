import numpy as np

from delayscan import TRclasses as tr
from delayscan.TRclasses import reduceBins

#INFO

# Instrument function parameters
# N0 * exp((-x)/l) + 1
IFparams = np.array([-6.90359945, 0.02964122])
# Low eKE cutoff at eKE = 0.126
IFcutoff = 0.126

# TOF to eKE calibration
# Flight tube length (m), time offset (ns), energy offset (eV)
calParams = np.array([0.62209127, 48.29925437, 1.10221363])

#Streaming potentials
SP_250618c = np.array([0.037811958986494684,  0.12997911802896056 ])
SP_250618d = np.array([-0.004002254614469857, 0.20421880748289375 ])

#Path to raw data
resultsPath = r"data/raw/"

#Info of rawdata
# resultsFolder,date, fn, pump wl, pulseEnergy pump,probe wl, pulseEnergy probe, integration time, approx time 0, SP before, SP after 
fileInfo_250618b = [
    (resultsPath,"180625","dt", 350.0, 5, 266.67, 7.4, 10, 71.7346, SP_250618c, SP_250618d),
    (resultsPath,"180625","dv", 350.0, 4, 266.67, 7.4, 10, 71.7346, SP_250618c, SP_250618d),
    (resultsPath,"180625","dx", 350.0, 4, 266.67, 7.4, 10, 71.7346, SP_250618c, SP_250618d),
    # (resultsPath,"180625","dz", 350.0, 4, 266.67, 7.4, 10, 71.7346, SP_250618c, SP_250618d),
    (resultsPath,"180625","eb", 350.0, 4, 266.67, 7.4, 10, 71.7346, SP_250618c, SP_250618d),
    (resultsPath,"180625","ec", 350.0, 4, 266.67, 7.4, 10, 71.7346, SP_250618c, SP_250618d),
    (resultsPath,"180625","eg", 350.0, 4, 266.67, 7.4, 10, 71.7346, SP_250618c, SP_250618d),
    (resultsPath,"180625","ei", 350.0, 4, 266.67, 7.4, 10, 71.7346, SP_250618c, SP_250618d),
    (resultsPath,"180625","ek", 350.0, 4, 266.67, 7.4, 10, 71.7346, SP_250618c, SP_250618d),
    (resultsPath,"180625","em", 350.0, 4, 266.67, 7.4, 10, 71.7346, SP_250618c, SP_250618d),
]

fileInfos = [fileInfo_250618b,]

#Rebins for final rebinned data
rebins = 10


# LOAD DATA

allDelays = []
print("Loading files")
for fileInfo in fileInfos:
    scanSet = tr.DelayScanSet()
    for info in fileInfo:

        scan = tr.DelayScan(resultsFolder = info[0],
                            date = info[1],
                            fileIndex = info[2],
                            pumpWavelength = info[3],
                            pumpPulseEnergy = info[4],
                            probeWavelength = info[5],
                            probePulseEnergy = info[6],
                            integrationTime = info[7],
                            timeZero = info[8],
                            SPbefore = info[9],
                            SPafter = info[10],
                            calParams = calParams,
                            IFparams = IFparams,
                            IFcutoff = IFcutoff,
                            )
        scan.importData()
        scan.applyJacobian()
        scan.convert_to_eKE()
        scan.applyInstrumentFunction()
        scan.correctStreamingPotential()
        scan.normaliseByIntegrationTime()
        print(f"- {info[1]}{info[2]}")

        scanSet.append(scan)
    allDelays.append(scanSet)
print("Files loaded")

print("\nProcessing data")

#SUBTRACT DATA
# Assuming all scans are the same
scan = scanSet[0]
# Finding the pump-probe signal by max counts
scan = scanSet[1]
counts = [np.sum(scan.spectra[i::4,1]) for i in range(4)]

pp_indices = np.argsort(counts)[::-1]
# Here the pump-only and probe-only signals could be the other way around
# but there is no way of telling the difference
# except for the shape of each spectrum
# As we are usually not so interested in pump-only
# and probe-only, it doesn't matter how we label them.

#Get subtracted spectra
for scan in scanSet:
    scan.subtract_background(*pp_indices)




# ADD SCANS
# 1. Remove data points below low eKE cutoff and above 5 eV
for scan in scanSet:
    scan.trim_energies()

# 2. Sum intensities for the same eKE and time delay (this can only be done if you didn't change the time points at any point & are using the same SP measurements)
scanSet.add_scans()

# 3. Rebin data if you like
scanSet.rebin_subtracted_spectra(rebins)
    
print("Data processed")

# SAVE DATA
print("\nSaving data")

savepath = r"data/out/"
# Save unbinned data
save_arr = tr.create_save_arr(scanSet.energies,
                              scanSet.delays,
                              scanSet.subtracted)

np.savetxt(savepath+"ExperimentalData.csv",save_arr, delimiter = ',',fmt = "%.10e")


# Save binned data
save_arr = tr.create_save_arr(scanSet.energies_rebinned,
                              scanSet.delays,
                              scanSet.subtracted_rebinned)

np.savetxt(savepath+"RebinnedExperimentalData.csv",save_arr, delimiter = ',',fmt = "%.10e")

print("Data saved")
