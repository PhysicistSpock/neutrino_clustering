from shared.preface import *


PHIs = 4
THETAs = 4
Vs = 4
NR_OF_NEUTRINOS = PHIs*THETAs*Vs
NU_MASS = 0.05*unit.eV

# Redshift integration parameters
#NOTE: When using LSODA method for solve_ivp with min_step, first_step
#NOTE: and max_step set, Z_AMOUNT effectively controls intrgration steps.
Z_START, Z_STOP, Z_AMOUNT = 0., 4., 50