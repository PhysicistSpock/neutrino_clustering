from shared.preface import *
import shared.my_units as myUnits
import shared.functions as fct


# Set initial vectors for each particle
y0 = np.ones(6)

# Solve EOMs
rho_0 = 1 
M_vir = 1

t = np.linspace(0,4,10)
trajectory = odeint(fct.EOMs, y0, t, args=(rho_0, M_vir))

# Fermi-Dirac value of final coords.

m_nu = 1  # neutrino mass
u_mu = [trajectory[-1,i]**2 for i in (3,4,5)]
p_nu = m_nu * u_mu
fct.Fermi_Dirac()