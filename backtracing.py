from matplotlib.pyplot import draw
from shared.preface import *
import shared.my_units as my
import shared.functions as fct
import shared.control_center as CC


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

    z_start, z_stop, z_amount = CC.Z_START, CC.Z_STOP, CC.Z_AMOUNT

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
    nu_Nr = CC.NR_OF_NEUTRINOS

    # Position of earth w.r.t Milky Way NFW halo center
    x1, x2, x3 = 8.5, 8.5, 0.
    x0 = np.array([x1, x2, x3])


    def draw_ui(amount):
        
        '''
        # conversion factor for limits
        cf = 5.3442883e-28 / CC.NU_MASS.to(unit.kg, unit.mass_energy()).value
        T_nu_eV = my.T_nu.to(unit.eV, unit.temperature_energy()).value
        
        # limits on velocity
        lower = 0.01*T_nu_eV*cf
        upper = 10*T_nu_eV*cf

        #? very confusing, limits way too high
        '''

        #! quick guesstimate
        lower = 2000
        upper = 4000

        # initial velocities array
        ui_km = np.geomspace(lower, upper, amount)*unit.km/unit.s
        ui_kpc = ui_km.to(unit.kpc/unit.s)

        # We drew magnitude of velocity, assume u_x=u_y=u_z.
        ui_array = np.array([np.ones(3)*(elem/np.sqrt(3)) for elem in ui_kpc])

        return ui_array


    # draw initial velocities
    ui = draw_ui(nu_Nr)

    # Combine vectors and append neutrino particle number
    y0_Nr = np.array([np.concatenate((x0,ui[i],[i+1])) for i in range(nu_Nr)])


    Processes = 16
    with ProcessPoolExecutor(Processes) as ex:
        ex.map(backtrack_1_neutrino, y0_Nr)  


    print('Execution time:', time.time()-start, 'seconds.')