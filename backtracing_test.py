from shared.preface import *
import shared.my_units as unit
import shared.functions as fct
start = time.time()


def EOMs(s, y, rho_0, M_vir):
    """Equations of motion for all x_i's and u_i's in terms of s."""


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

    #TODO: explore interpolation function for in between zeds

    # derivative of grav. potential in [kpc/s**2]
    derivative_vector = fct.dPsi_dxi_NFW(x_i, z[0], rho_0, M_vir)

    # global minus sign for dydt array, s.t. calculation is backwards in time
    dyds = -np.array([u_i, -(1+z)**-2 * derivative_vector])
    
    #NOTE on units:
    # dydt array will be in [[kpc/s],[kpc/s**s]]
    # and after integration in [[kpc],[kpc/s]], correct for x_i and u_i resp.

    # reshape from (2,3) to (6,), s.t. the vector looks like
    # (x_1, x_2, x_3, u_1, u_2, u_3), required by solve_ivp algorithm
    return np.reshape(dyds, 6)


def backtrack_1_neutrino():
    ...

if __name__ == '__main__':

    #TODO: What is the coordinate origin? Is it w.r.t the NFW halo radius at the
    #      position of earth?

    x1, x2, x3 = 0, 0, 10.
    u1, u2, u3 = 0., 0., 0.1

    # initial spatial positions in [kpc] and velocities in [kpc/s]
    Xunit, Uunit = unit.kpc, unit.kpc/unit.s
    x0 = np.array([x1, x2, x3]) * Xunit
    u0 = np.array([u1, u2, u3]) * Uunit

    # combined inital vector
    y0 = np.array([x0, u0]).flatten()

    # NFW halo parameters
    rho_0 = unit.rho0_NFW 
    M_vir = unit.Mvir_NFW


    # Redshifts to integrate over
    zeds = np.linspace(0,0.5,50)

    # Array to store solutions
    sols = []

    loop = range(len(zeds)-1)
    for zi in loop:

        # Append initial phase-space vector to solutions array
        if zi == loop[0]:
            sols.append(y0)

        # Redshift and converted to time variable s
        z0, z1 = zeds[zi], zeds[zi+1]
        z_steps = np.array([z0, z1])
        s_steps = np.array([fct.s_of_z(z0), fct.s_of_z(z1)])  # in [s] already

        # Solve all 6 EOMs
        sol = solve_ivp(EOMs, s_steps, y0, args=(rho_0, M_vir))

        # Overwrite current vector with new one (already has Xunit and Uunit).
        y0 = np.array([sol.y[0:3,-1], sol.y[3:6,-1]]).flatten()

        # Append last phase-space vector to solutions array
        if zi == loop[-1]:
            sols.append(y0)


    print('Solution array shape:', np.array(sols).shape)

    '''
    with np.printoptions(precision=12):

        change_xi = (new_xi-x0) / Xunit
        print('change_xi:', change_xi)

        change_ui = (new_ui-u0) / (unit.m/unit.s)
        print('change_ui', change_ui, 'new_ui:', new_ui/Uunit)
    '''



    """
    # Fermi-Dirac value of final coords.
    m_nu = 1  # neutrino mass
    u_mu = np.sum(new_ui**2)
    p_nu = m_nu * u_mu
    FDval = fct.Fermi_Dirac(p_nu)
    """

    print('Execution time:', time.time()-start, 'seconds.')