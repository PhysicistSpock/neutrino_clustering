from shared.preface import *


PHIs = 5
THETAs = 3
Vs = 3
NR_OF_NEUTRINOS = PHIs*THETAs*Vs
NU_MASS = 0.05*unit.eV

# Redshift integration parameters
Z_START, Z_STOP, Z_AMOUNT = 0., 4., 100

# Control if simulation runs forwards (+1) or backwards (-1) in time. 
TIME_FLOW = -1