from shared.preface import *
import shared.my_units as unit
import shared.functions as fct


# Set initial vectors for particle and constants used
x1, x2, x3 = 1, 1, 1
u1, u2, u3 = 1, 1, 1
x0 = np.array([x1, x2, x3])
u0 = np.array([u1, u2, u3])
y0 = np.array([x0, u0]).flatten()

rho_0 = unit.rho0_NFW 
M_vir = unit.Mvir_NFW
zeds = np.linspace(1,4,3)

# calculate grav. potential derivatives
derivative_vector = fct.dPsi_dxi_NFW(x0, zeds[0], rho_0, M_vir)

# Convert redshift z to time variable s.
s = np.array([fct.s_of_z(zeds[0]),fct.s_of_z(zeds[1])])

# Solve EOMs 
sol = odeint(fct.EOMs, y0, s, args=(rho_0, M_vir))

# New position and velocity vectors.
next_xi = np.array([sol[-1,0:3]]).flatten()
next_ui = np.array([sol[-1,3:6]]).flatten()
print(next_xi-x0)
print(next_ui-u0)


'''
# Fermi-Dirac value of final coords.
m_nu = 1  # neutrino mass
u_mu = np.sum(next_ui**2)
p_nu = m_nu * u_mu
FDval = fct.Fermi_Dirac(p_nu)
'''