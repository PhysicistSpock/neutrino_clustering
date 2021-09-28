from shared.preface import *


T_nu      = 1.95*unit.K                # Relic neutrino temp. today

# Cosmology - Mertsch et al. (2020)
Omega_m0  = 0.3111                     # Matter-energy density today
Omega_L0  = 1 - Omega_m0               # Dark-energy density today

H_UNIT    = unit.km/unit.s/unit.Mpc
H0        = 67.66*H_UNIT               # Hubble constant today
h         = H0/100/H_UNIT              # Dimensionless Hubble constant

# NFW parameters today - Mertsch et al. (2020)
Mvir_NFW  = 2.03e12*unit.M_sun      # Virial mass
rho0_NFW  = 1.06e7*(unit.M_sun/unit.kpc**3)  # density normalization
r_s_NFW   = 19.9*unit.kpc                   # scale radius 
r_vir_NFW = 333.5*unit.kpc                  # virial radius


# Units for initial spatial positions and velocities.
Xunit, Uunit = unit.kpc, unit.kpc/unit.s