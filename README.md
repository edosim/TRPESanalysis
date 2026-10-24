# TRPESanalysis

A Python toolkit for processing and analysing liquid-microjet ultraviolet time-resolved photoelectron spectroscopy (TRPES) data recorded at the Ultrafast Laser Facility, University College London.

It takes raw time-of-flight (TOF) photoelectron spectra and produces energy-calibrated, background-subtracted pump-probe sets of photoelectron spectra. It then fits kinetic models and corrects for inelastic electron scattering in the liquid.

## Features

- **Data processing pipeline.** Loads every spectrum in a delay scan and applies, in order:
  - the Jacobian for the conversion of TOF to electron kinetic energy (eKE)
  - the TOF to eKE calibration
  - the correction for the lower collection efficiency of the spectrometer at low eKEs (instrument function)
  - the correction for electrokinetic charging of the liquid jet (streaming potential)
  - integration time normalisation
- **Background subtraction.** Separates pump–probe, pump-only, probe-only and background spectra at each delay and extracts the background-free pump–probe signal.
- **Scan aggregation.** Sums repeated scans and performs energy trimming and rebinning.
- **Time-zero determination.** Fits total counts with exponentials convolved with a Gaussian instrument response function (IRF).
- **Spectral retrieval.** Recovers undistorted spectra by fitting sums of Gaussians through scattering matrices from Monte Carlo simulations (see the repo [LJscatter](https://github.com/FieldingGroup/LJscatter/)).
- **Global kinetic fitting.** Fits full 2D energy vs. delay datasets with `scipy.optimize.curve_fit`, using:
  - **EAS**: evolution-associated spectra with the sequential model A → B → C
  - **DAS**: decay-associated spectra with two parallel exponential decays
  - **Retrieved EAS**: the EAS model fited with Gaussians distorted by scattering, giving undistorted spectra.

## Repository structure
```
TRPESanalysis/
├── trpes/              # Core package
│   ├── delayscan.py        # DelayScan and DelayScanSet: loading, conversions and corrections, subtraction, aggregation
│   ├── conversions.py      # TOF to eKE, stage position to delay, instrument function
│   ├── kineticmodels.py    # EAS, DAS and retrieved EAS global kinetic models
│   └── retrieve.py         # Scattering-basis retrieval (JetFitFunc), rebinning, Gaussian functions
├── TRPESanalysis.ipynb     # Data processing, time zero, spectral retrieval and plots
├── KineticFitting.ipynb    # EAS/DAS global fits, residuals and retrieved-Gaussian kinetics and plots
├── assets/TRPES.mplstyle   # Matplotlib style for figures
├── bases/                  # Scattering basis sets (HDF5, not tracked)
└── data/
    ├── raw/                # Raw TOF spectra (not tracked)
    └── out/                # Processed CSV output (not tracked)
```

## Requirements

- Python 3.11+
- NumPy, SciPy, Matplotlib, h5py, Jupyter

Create and activate the environment with conda:

```bash
conda env create -f environment.yml
conda activate trpes
```

## Data

Raw data and processed output are not included because the results are unpublished. The basis sets are not included because of their size (~1.7 GB) and are available on request.

Raw data should be placed in `data/raw/` using the UCL folder layout:
 
```
data/raw/<date>/<date><fileIndex>/<date><fileIndex>/<delay>_<...>.txt
```

There is one text file per spectrum. The first undescore-separated field is the delay stage position at which it was recorded in mm. There should be four files per delay stage position: one for each of pump-probe, pump-only, probe-only and background spectra. Files are read in order of modification time cycling through the four spectra at each delay.

THe basis sets for spectral retrieval (`basisSets_pCE_2026.hdf5`) go in `bases/`. They contain matrices mapping initial to finela eKE obtained from Monte Carlo electron-scattering simulations on the grid `np.arange(0.01, 5.01, 0.01)` eV (see the repo [LJscatter](https://github.com/FieldingGroup/LJscatter/)).

## Usage

1. Add the data to `data/raw/` and the basis set to `bases/`
2. Open `TRPESanalysis.ipynb`. set the calibration constants, the streaming potential correction the metadata of each scan to load. Then run all cells. The notebook processes the raw spectra, finds time zero, retrieves the scattering-corrected spectra at each time delay and writes the results to `data/out`
3. Run `KineticFitting.ipynb` to fit the EAS, DAS and retrieved EAS models. The notebook prints the fit results, and plots the fits and residuals.

## Output files
 
 ```
| File | Contents |
|---|---|
| `ExperimentalData.csv`         # Summed, background-subtracted spectra (eKE × delay as first row and first column) 
| `RebinnedExperimentalData.csv` # Same, rebinned along eKE
| `SubtractedSpectra.csv`        # Rebinned spectra with delays shifted by fitted time zero
| `LJscatterGridData.csv`        # Spectra rebinned onto the eKE grid used required by `basis`
| `RetrievedData.csv`            # Spectra retrieved at each delay
```


## License
 
MIT. See [LICENSE](LICENSE).
