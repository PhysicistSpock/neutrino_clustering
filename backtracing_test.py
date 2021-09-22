from shared.preface import *
import shared.my_units as unit
import shared.functions as fct

#TODO: What is the coordinate origin? Is it w.r.t the NFW halo radius at the
#      position of earth?

x1, x2, x3 = 1, 1, 1
u1, u2, u3 = 1, 2, 3

# initial spatial positions in [kpc] and velocities in [kpc/s]
x0 = np.array([x1, x2, x3]) * unit.kpc
u0 = np.array([u1, u2, u3]) * unit.kpc/unit.s

# combined inital vector
y0 = np.array([x0, u0]).flatten()

# NFW halo parameters
rho_0 = unit.rho0_NFW 
M_vir = unit.Mvir_NFW



#
### Solve EOMs ###
#

# redshifts to integrate over
zeds = np.linspace(0,4,100)

# Redshift and converted to time variable s
z0, z1 = zeds[0], zeds[1]
z_steps = np.array([z0, z1])
s_steps = np.array([fct.s_of_z(z0), fct.s_of_z(z1)])  # in [s] already


def EOMs(s, y, rho_0, M_vir):

    # initialize vector: x_i in [kpc], u_i in [kpc/s]
    x_i, u_i = np.reshape(y, (2,3))

    # Pick out redshift z according to current time s
    #NOTE: solve_ivp algorithm calculates steps between s_steps, for these we
    #      just use the z value for the starting s
    z_for_between_s_steps = 0.
    if s in s_steps:
        z = z_steps[s_steps==s]
        z_for_between_s_steps = z
    else:
        z = np.array([z_for_between_s_steps])

    # derivative of grav. potential in [kpc/s**2]
    derivative_vector = fct.dPsi_dxi_NFW(x_i, z[0], rho_0, M_vir)

    # global minus sign for dydt array, s.t. calculation is backwards in time
    dydt = -np.array([u_i, -(1+z)**-2 * derivative_vector])
    
    # Note on units:
    # dydt array will be in [[kpc/s],[kpc/s**s]]
    # and after integration in [[kpc],[kpc/s]], correct for x_i and u_i resp.

    # reshape from (2,3) to (6,), s.t. the vector looks like
    # (x_1, x_2, x_3, u_1, u_2, u_3), required by solve_ivp algorithm
    return np.reshape(dydt, 6)


sol = solve_ivp(EOMs, s_steps, y0, args=(rho_0, M_vir))


# New position and velocity vectors.
next_xi = np.array([sol.y[0:3,-1]]).flatten()
next_ui = np.array([sol.y[3:6,-1]]).flatten()
print(next_xi-x0)
print(next_ui-u0)

num_of_non_zeros = np.count_nonzero(next_ui-u0)
if num_of_non_zeros == 0:
    print('Array contains only 0')
else:
    print('Array has non-zero items')


"""
# Fermi-Dirac value of final coords.
m_nu = 1  # neutrino mass
u_mu = np.sum(next_ui**2)
p_nu = m_nu * u_mu
FDval = fct.Fermi_Dirac(p_nu)
"""