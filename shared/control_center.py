from shared.preface import *

PHIs = 10
THETAs = 10
Vs = 100
NR_OF_NEUTRINOS = PHIs*THETAs*Vs

LOWER = 0.01
UPPER = 10.
MOMENTA = np.geomspace(LOWER, UPPER, Vs)

NU_MASS = 0.01*unit.eV
NU_MASS_KG = NU_MASS.to(unit.kg, unit.mass_energy())
N0 = 112  # standard neutrino number density in [1/cm**3]

## Redshift integration parameters
#NOTE: Linearly spaced, denser for late times (closer to today)
late_steps = 200
early_steps = 100
Z_START, Z_STOP, Z_AMOUNT = 0., 4., late_steps+early_steps
z_late = np.linspace(0,2,200)
z_early = np.linspace(2.01,4,100)
ZEDS = np.concatenate((z_late, z_early))

# Control if simulation runs forwards (+1) or backwards (-1) in time. 
TIME_FLOW = -1

SOLVER = 'RK23'