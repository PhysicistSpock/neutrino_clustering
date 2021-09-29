from numpy.random import random
from shared.preface import *
import shared.my_units as my
import shared.functions as fct


def EOMs(s, y, rho_0, M_vir):
    """Equations of motion for all x_i's and u_i's in terms of s."""

    # initialize vector and bestow units
    x_i_vals, u_i_vals = np.reshape(y, (2,3))
    x_i, u_i = x_i_vals*my.Xunit, u_i_vals*my.Uunit

    # Pick out redshift z according to current time s
    #NOTE: solve_ivp algorithm calculates steps between s_steps, for these we
    #      just use the z value for the starting s
    z_for_between_s_steps = 0.
    if s in s_steps:
        z = z_steps[s_steps==s]
        z_for_between_s_steps = z
    else:
        z = np.array([z_for_between_s_steps])

    derivative_vector = fct.dPsi_dxi_NFW(x_i, z[0], rho_0, M_vir)

    # global minus sign for dydt array, s.t. calculation is backwards in time
    u_i_kpc = u_i.to(unit.kpc/unit.s)
    dyds = -np.array([u_i_kpc.value, -(1+z)**-2 * derivative_vector.value])

    #NOTE: reshape from (2,3) to (6,), s.t. the vector looks like
    #      (x_1, x_2, x_3, u_1, u_2, u_3), required by solve_ivp algorithm
    return np.reshape(dyds, 6)


def backtrack_1_neutrino(y0_Nr):
    """Simulate trajectory of 1 neutrino."""

    #! Redshift start, redshift to integrate back to, redshift amount of steps
    z_start, z_stop, z_amount = 0, 0.5, 5

    global z_steps, s_steps  # other functions can use these variables

    # Split input into initial vector and neutrino number
    y0, Nr = y0_Nr[0:-1], y0_Nr[-1]

    # Redshifts to integrate over
    zeds = np.linspace(z_start, z_stop, z_amount)

    
    sols = []
    loop = range(len(zeds)-1)
    for zi in loop:

        # Save initial phase-space vector
        if zi == loop[0]:
            sols.append(y0)

        # Redshift and converted time variable s
        z0, z1 = zeds[zi], zeds[zi+1]
        z_steps = np.array([z0, z1])
        s_steps = np.array([fct.s_of_z(z0), fct.s_of_z(z1)])

        # Solve all 6 EOMs
        #NOTE: output as raw numbers but in [kpc, kpc/s]
        sol = solve_ivp(EOMs, s_steps, y0, args=(my.rho0_NFW, my.Mvir_NFW))

        # Overwrite current vector with new one.
        y0 = np.array([sol.y[0:3,-1], sol.y[3:6,-1]]).flatten()

        # Save last phase-space vector
        if zi == loop[-1]:
            sols.append(y0)

    np.save(f'neutrino_vectors/nu_{int(Nr)}.npy', np.array(sols))



if __name__ == '__main__':
    start = time.time()

    #! Amount of neutrinos to simulate
    neutrinos = 2

    # Position of earth w.r.t Milky Way NFW halo center
    x1, x2, x3 = 8.5, 8.5, 0.
    x0 = np.array([x1, x2, x3])
    
    # Random draws for velocities
    ui_min, ui_max = 2000.*unit.km/unit.s, 4000.*unit.kpc/unit.s
    ui_min_kpc, ui_max_kpc = ui_min.to(my.Uunit), ui_max.to(my.Uunit)
    ui = np.array([
        np.random.default_rng().uniform(ui_min_kpc.value, ui_max_kpc.value, 3) 
        for _ in range(neutrinos)
        ])

    # Combine vectors and append neutrino particle number
    y0_Nr = np.array([np.concatenate((x0,ui[i],[i+1])) for i in range(neutrinos)])


    Processes = 1
    with ProcessPoolExecutor(Processes) as ex:
        ex.map(backtrack_1_neutrino, y0_Nr)  

    #
    ### Calculate number density
    #

    # neutrino mass
    m_nu = 0.05 * unit.eV  # in natural units
    m_nu_kg = m_nu.to(unit.kg, unit.mass_energy())  # in SI units

    p0s, p_backs = np.zeros(neutrinos), np.zeros(neutrinos)
    for Nr in range(neutrinos):
        
        # load initial velocity -> momentum today
        u0 = np.load(f'neutrino_vectors/nu_{int(Nr+1)}.npy')[0][3:6]
        p0 = np.sqrt(np.sum(u0**2)) * m_nu_kg.value
        p0s[Nr] = p0

        # load "last" velocity -> momentum at z_back
        u_back = np.load(f'neutrino_vectors/nu_{int(Nr+1)}.npy')[-1][3:6]
        p_back = np.sqrt(np.sum(u_back**2)) * m_nu_kg.value
        p_backs[Nr] = p_back


    #NOTE: Attach units [kg*kpc/s] to p0s and p_backs.
    p_unit = unit.kg*unit.kpc/unit.s
    p0s, p_backs = p0s*p_unit, p_backs*p_unit

    n_nu = fct.number_density(p0s, p_backs, m_nu)
    # print('Number density:', fct.number_density(p0, p_back))


    print('Final number density:', n_nu)


    print('Execution time:', time.time()-start, 'seconds.')