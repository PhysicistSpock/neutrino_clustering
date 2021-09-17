from shared.preface import *

def rho_NFW(r, rho_0, r_s):
    """NFW density profile

    Args:
        r (array): radius from center
        rho_0 (array): normalisation 
        r_s (array): scale radius

    Returns:
        array: density at radius r
    """    

    rho = rho_0 / (r/r_s) / np.power(1+(r/r_s), 2)
    return rho


def concentration_param(z):
    