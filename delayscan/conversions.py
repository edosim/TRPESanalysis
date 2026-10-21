
import re
import numpy as np

### Constants
MASS_ELECTRON = 0.51099895e6 # eV
LIGHTSPEED = 299_792_458e-9 #m/ns and mm/ps

def tof_to_eke (t,s,t0,E0):
    """Convert electron time-of-flight (TOF) to electron kinetic energy (eKE).
    
    Parameters
    --------
    t : numpy.ndarray
        Array of TOF values to convert.
    s : float
        Length of flight tube in m.
    t0 : float
        Time offset (between laser trigger and when the laser 
        reaches the interaction region).
    E0 : float
        Energy offset (Difference in potential between the 
        interaction region and the detector).

    Returns
    ------
    scalar or numpy.ndarray
        eKE value of `t`.
    """
    return (0.5* MASS_ELECTRON/(LIGHTSPEED**2)) * (s/(t-t0))**2 - E0

def instrument_function(x,N0,l):
    """Models reduced collection efficiency of the spectrometer for electrons
    with eKE ~ 0 eV with the function::

        N0 * e^(-x/l) + 1.

    
    Parameters
    -------------
    x : numpy.ndarray
        Array of eKEs.
    N0 : float
        Pre-exponential factor.
    l : float
        decay constant.

    Returns
    -----------
    scalar or numpy.ndarray
        Collection efficiency of spectrometer at ``x``.

    """
    return N0*np.exp((-x)/l)+1


def pos_to_time(positions, pos0):
    """Convert the delay stage position (in mm) relative to the time-zero position
    to time delay (in ps). Includes the double pass on the delay stage.
    
    Parameters
    ------------
    positions : numpy.ndarray
        Array of delay stage positions in mm.
    pos0 : float
        Position of time zero in mm.

    Returns
    -----------
    scalar or numpy.ndarray
        Time delay of ``positions``.
    """
    return 2*(positions-pos0)/(LIGHTSPEED)

def delaystring_to_float(delay_string):
    """Convert delay string of the form "XXpYYYY" where "XX" and "YYYY""
    represent the digits before and after the decimal point, respectively,
    to the float "XXpYYYY".
    
    Parameters
    -------------
    delay_string : str
        String of the form "XXpYYYY".

    Returns
    -------------
    float
        Float representation of ``delay_string``.
    """
    #Split by letter. Returns only numbers.
    num = re.split('[a-zA-Z]', delay_string)
    num = float(num[0]) + float(num[1])*1e-4
    return num


def delaystrings_to_float(delay_strings):
    """Convert a sequence of delay strings of the form "XXpYYYY" where "XX" and "YYYY"
    represent the digits before and after the decimal point, respectively,
    to the floats "XX.YYYY".
    
    Parameters
    -------------
    delay_strings : sequence of str
        Sequence of strings of the form "XXpYYYY".

    Returns
    -------------
    np.ndarray
        Array of the float representations of ``delay_strings``.
    """
    return np.array([delaystring_to_float(s) for s in delay_strings])

