from shared.preface import *


# Energy 
GeV       = 1.0                        # Unit of energy: GeV
eV        = GeV/1.0e9
keV       = 1.0e3*eV
MeV       = 1.0e3*keV
TeV       = 1.0e3*GeV
erg       = TeV/1.602                  # erg
J         = 1.0e7*erg                  # joule
   
# Length
cm        = 5.0678e4/eV                # centimeter
m         = 1.0e2*cm
km        = 1.0e3*m
pc        = 3.086e18*cm                # parsec
kpc       = 1.0e3*pc
Mpc       = 1.0e3*kpc

# Time
s         = 2.9979e10*cm               # second
   
# Mass
kg        = J/m**2*s**2
gram      = kg/1000.
Msun      = 1.989e30*kg                # Mass of the Sun
G         = 6.674e-11*m**3/kg/s**2     # Gravitational constant

# Angle
deg       = np.pi/180.0                # Degree
arcmin    = deg/60.                    # Arcminute
arcsec    = arcmin/60.                 # Arcsecond
sr        = 1.                         # Steradian

# Temperature
K         = 1.380649e-23*J             # Kelvin
kb        = 1.                         # Boltzmann constant
Tnu       = 1.95*K                     # Relic neutrino temp. today

# Cosmology
Omega_m0  = 0.3111                     # Matter-energy density today
Omega_L0  = 1 - Omega_m0               # Dark-energy density today
H0        = 67.66*km/s/Mpc             # Hubble constant today
h         = H0/(100*km/s/Mpc)          # Dimensionless Hubble constant

# NFW parameters today - Mertsch et al. (2020)
Mvir_NFW  = 2.03e12*Msun               # Virial mass
rho0_NFW  = 1.06e7*Msun/kpc**3         # density normalization
r_s_NFW   = 19.9*kpc                   # scale radius 
r_vir_NFW = 333.5*kpc                  # virial radius



### Test units all 1
"""
# Energy 
GeV       = 1.
eV        = 1.
keV       = 1.
MeV       = 1.
TeV       = 1.
erg       = 1.
J         = 1.
   
# Length
cm        = 1.
m         = 1.
km        = 1.
pc        = 1.
kpc       = 1.
Mpc       = 1.

# Time
s         = 1.
   
# Mass
kg        = 1.
gram      = 1.
Msun      = 1.
G         = 1.

# Angle
deg       = 1.
arcmin    = 1.
arcsec    = 1.
sr        = 1.

# Temperature
K         = 1.
kb        = 1.
Tnu       = 1.

# Cosmology
Omega_m0  = 1.
Omega_L0  = 1.
H0        = 1.
h         = 1.

# NFW parameters today - Mertsch et al. (2020)
Mvir_NFW  = 1.
rho0_NFW  = 1.
r_s_NFW   = 1.
r_vir_NFW = 1.
"""