from shared.preface import *


PHIs = 5
THETAs = 5
Vs = 100
NR_OF_NEUTRINOS = PHIs*THETAs*Vs
NU_MASS = 0.05*unit.eV
NU_MASS_KG = NU_MASS.to(unit.kg, unit.mass_energy())

# Redshift integration parameters
Z_START, Z_STOP, Z_AMOUNT = 0., 4., 100

# Control if simulation runs forwards (+1) or backwards (-1) in time. 
TIME_FLOW = -1

# Integration method
SOLVER = 'LSODA'