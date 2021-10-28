from shared.preface import *


PHIs = 10
THETAs = 10
Vs = 100
NR_OF_NEUTRINOS = PHIs*THETAs*Vs
NU_MASS = 0.05*unit.eV
NU_MASS_KG = NU_MASS.to(unit.kg, unit.mass_energy())
N0 = 112  # standard neutrino number density in [1/cm**3]

# Redshift integration parameters
Z_START, Z_STOP, Z_AMOUNT = 0., 4., 99
Z_START_LOG = 1e-1
zeds_pre = np.geomspace(Z_START_LOG, Z_STOP, Z_AMOUNT) - Z_START_LOG
ZEDS = np.insert(zeds_pre, len(zeds_pre), 4.)

# Control if simulation runs forwards (+1) or backwards (-1) in time. 
TIME_FLOW = -1

# Integration method
SOLVER = 'LSODA'