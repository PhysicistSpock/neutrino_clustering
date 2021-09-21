from shared.preface import *
import shared.my_units as myUnits
import shared.functions as fct


# Set initial vectors for particle and constants used
x0 = np.ones(3)
u0 = np.array([2,3,4])
y0 = np.array([x0, u0]).flatten()

rho_0 = 1 
M_vir = 1
zeds = np.linspace(0,4,10)

# calculate grav. potential derivatives
derivative_vector = fct.dPsi_dxi_NFW(x0, zeds[0], rho_0, M_vir)

# Convert redshift z to time variable s.
s = np.array([fct.s_of_z(zeds[0]),fct.s_of_z(zeds[1])])

# Solve EOMs 
sol = odeint(fct.EOMs, y0, s, args=(rho_0, M_vir))

# New position and velocity vectors.
next_xi = np.array([sol[-1,0:3]]).flatten()
next_ui = np.array([sol[-1,3:6]]).flatten()

print(next_xi)
print(next_ui)


'''
# Fermi-Dirac value of final coords.
m_nu = 1  # neutrino mass
u_mu = np.sum(next_ui**2)
p_nu = m_nu * u_mu
FDval = fct.Fermi_Dirac(p_nu)
'''