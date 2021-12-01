from shared.preface import *

PHIs = 20
THETAs = 20
Vs = 100
NR_OF_NEUTRINOS = PHIs*THETAs*Vs

LOWER = 0.01
UPPER = 40.

NU_MASS = 0.05*unit.eV
NU_MASS_KG = NU_MASS.to(unit.kg, unit.mass_energy())
N0 = 112  # standard neutrino number density in [1/cm**3]

# Redshift integration parameters

#NOTE: OLD, logspaced
# Z_START, Z_STOP, Z_AMOUNT = 0., 4., 199
# Z_START_LOG = 1e-1
# zeds_pre = np.geomspace(Z_START_LOG, Z_STOP, Z_AMOUNT) - Z_START_LOG
# ZEDS = np.insert(zeds_pre, len(zeds_pre), 4.)

#NOTE: NEW, linear spaced, denser for late times (closer to today)
late_steps = 200
early_steps = 100
Z_START, Z_STOP, Z_AMOUNT = 0., 4., late_steps+early_steps
z_late = np.linspace(0,2,200)
z_early = np.linspace(2.01,4,100)
ZEDS = np.concatenate((z_late, z_early))

# Control if simulation runs forwards (+1) or backwards (-1) in time. 
TIME_FLOW = -1

SOLVER = 'RK45'  #! run sim with 'RK45' like they did