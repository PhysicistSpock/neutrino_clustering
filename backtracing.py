from numpy.random import random
from shared.preface import *
import shared.my_units as unit
import shared.functions as fct


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


def backtrack_1_neutrino(y0_Nr):
    global z_steps, s_steps

    # Split input into initial vector and neutrino number
    y0, Nr = y0_Nr[0:-1], y0_Nr[-1]

    # NFW halo parameters
    rho_0 = unit.rho0_NFW 
    M_vir = unit.Mvir_NFW

    # Redshifts to integrate over
    zeds = np.linspace(0,0.5,10)

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
        
        if zi in (0,1):
            print('Solve EOMs:', time.time()-start, 'seconds.')

        # Overwrite current vector with new one (already has Xunit and Uunit).
        y0 = np.array([sol.y[0:3,-1], sol.y[3:6,-1]]).flatten()

        # Append last phase-space vector to solutions array
        if zi == loop[-1]:
            sols.append(y0)

    np.save(f'neutrino_vectors/nu_{int(Nr)}.npy', np.array(sols))
    # print(f'nu_{int(Nr)} vector:', np.array(sols)[-1])

if __name__ == '__main__':
    start = time.time()

    # Initial spatial positions in [kpc] and velocities in [kpc/s]
    Xunit, Uunit = unit.kpc, unit.kpc/unit.s

    # Position of earth w.r.t Milky Way NFW halo center
    x1, x2, x3 = 8.5, 8.5, 0.
    x0 = np.array([x1, x2, x3]) * Xunit
    
    # Random draws for velocities
    ui_min, ui_max, ui_size = 0.1, 1., 1
    ui = np.array([
        np.random.default_rng().uniform(ui_min, ui_max, 3) 
        for _ in range(ui_size)
        ]) * Uunit

    # Combine vectors and append neutrino particle number
    y0_Nr = np.array([np.concatenate((x0,ui[i],[i+1])) for i in range(ui_size)])


    with ThreadPoolExecutor(os.cpu_count()*2) as ex:
        ex.map(backtrack_1_neutrino, y0_Nr)  

    #
    ### Calculate number density
    #

    m_nu = 1.  # neutrino mass
    n_nu = 0.
    for Nr in range(ui_size):
        ui = np.load(f'neutrino_vectors/nu_{int(Nr+1)}.npy')[-1][3:6]
        pi = np.sum(ui**2) * m_nu

        n_nu += fct.number_density(pi)
    
    print(n_nu)


    print('Execution time:', time.time()-start, 'seconds.')