
import re
import numpy as np

### Constants
MASS_ELECTRON = 0.51099895e6 # eV
LIGHTSPEED = 299_792_458e-9 #m/ns and mm/ps

def tof_to_eke (t,s,t0,E0):
    """Convert data from TOF to eKE"""
    return (0.5* MASS_ELECTRON/(LIGHTSPEED**2)) * (s/(t-t0))**2 - E0


def pos_to_time(delays, pos0):
    """Convert delays from mm to ps"""
    return 2*(delays-pos0)/(LIGHTSPEED)

def delaystring_to_float(delay_string):
    """Convert delay string in file name (XXpYYYY) to float (XX.YYYY)"""
    #Split by letter. Returns only numbers.
    num = re.split('[a-zA-Z]', delay_string)
    num = float(num[0]) + float(num[1])*1e-4
    return num


def delaystrings_to_float(delay_strings):
    """Convert list of delay strings in file names to floats"""
    return np.array([delaystring_to_float(s) for s in delay_strings])


