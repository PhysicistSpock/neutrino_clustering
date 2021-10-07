from matplotlib.pyplot import draw
from shared.preface import *
import shared.my_units as my
import shared.functions as fct
import shared.control_center as CC


def draw_ui(v_points, phi_points, theta_points):
    """Get initial velocities for the neutrinos."""
    
    # conversion factor for limits
    cf = 5.3442883e-28 / CC.NU_MASS.to(unit.kg, unit.mass_energy()).value
    T_nu_eV = my.T_nu.to(unit.eV, unit.temperature_energy()).value
    
    # limits on velocity
    lower = 0.01*T_nu_eV*cf * unit.m/unit.s
    upper = 10*T_nu_eV*cf * unit.m/unit.s

    # convert to km/s
    low, up = lower.to(unit.km/unit.s).value, upper.to(unit.km/unit.s).value

    # Initial magnitudes of the velocities
    v_km = np.geomspace(low, up, v_points)*unit.km/unit.s
    v_kpc = v_km.to(unit.kpc/unit.s).value

    # Split up this magnitude into velocity components
    #NOTE: done by using spher. coords. trafos, which act as "weights"

    eps = 0.01  # shift in theta, so poles are not included
    ps = np.linspace(0., 2.*np.pi, phi_points)
    ts = np.linspace(0.+eps, np.pi-eps, theta_points)

    uxs = [v*np.cos(p)*np.sin(t) for t in ts for p in ps for v in v_kpc]
    uys = [v*np.sin(p)*np.sin(t) for t in ts for p in ps for v in v_kpc]
    uzs = [v*np.cos(t) for t in ts for p in ps for v in v_kpc]

    ui_array = np.array([[ux, uy, uz] for ux,uy,uz in zip(uxs,uys,uzs)])        

    return ui_array 


def EOMs(s, y, rho_0, M_vir):
    """Equations of motion for all x_i's and u_i's in terms of s."""

    # initialize vector and bestow units
    x_i_vals, u_i_vals = np.reshape(y, (2,3))
    x_i, u_i = x_i_vals*my.Xunit, u_i_vals*my.Uunit

    # Pick out redshift z according to current time s
    #NOTE: for in between s_steps (done by solve_ivp) we take initial redshift
    if s in s_steps:
        z = z_steps[s_steps==s][0]
    else:
        z = z_steps[0]

    gradient = fct.dPsi_dxi_NFW(x_i, z, rho_0, M_vir)

    u_i_kpc = u_i.to(unit.kpc/unit.s)
    dyds = [u_i_kpc.value, -(1+z)**-2 * gradient.value]
    dyds = np.reshape(dyds, (6,))

    return dyds


def backtrack_1_neutrino(y0_Nr):
    """Simulate trajectory of 1 neutrino."""

    global z_steps, s_steps, Nr  # so other functions can use these variables

    # Split input into initial vector and neutrino number
    y0, Nr = y0_Nr[0:-1], y0_Nr[-1]

    # Redshifts to integrate over
    # zeds = np.linspace(CC.Z_START, CC.Z_STOP, CC.Z_AMOUNT)  # linear
    zeds = np.geomspace(1e-10, CC.Z_STOP, CC.Z_AMOUNT)  # log

    # solutions array with initial and final vector for 1 neutrino
    sols = []
    sols.append(y0)  # save initial vector

    for zi in range(len(zeds)-1):

        # Redshift and converted time variable s
        z0, z1 = zeds[zi], zeds[zi+1]
        z_steps = np.array([z0, z1])
        s_steps = np.array([fct.s_of_z(z0), fct.s_of_z(z1)])     

        # Solve all 6 EOMs
        #NOTE: output as raw numbers but in [kpc, kpc/s]
        sol = solve_ivp(
            EOMs, s_steps, y0, method='RK45',
            args=(my.rho0_NFW, my.Mvir_NFW)
            )

        # Overwrite current vector with new one.
        y0 = np.array([sol.y[0:3,-1], sol.y[3:6,-1]]).flatten()

        sols.append(y0)  # save current vector

    np.save(f'neutrino_vectors/nu_{int(Nr)}.npy', np.array(sols))


if __name__ == '__main__':
    start = time.time()

    #! Amount of neutrinos to simulate
    nu_Nr = CC.NR_OF_NEUTRINOS

    # Position of earth w.r.t Milky Way NFW halo center.
    #NOTE: Earth is placed on x axis of coord. system.
    x1, x2, x3 = 8.5, 0., 0.
    x0 = np.array([x1, x2, x3])

    # Draw initial velocities.
    ui = draw_ui(CC.PHIs, CC.THETAs, CC.Vs)
    
    # Combine vectors and append neutrino particle number.
    y0_Nr = np.array([np.concatenate((x0,ui[i],[i+1])) for i in range(nu_Nr)])

    # Run simulation on multiple cores.
    Processes = 16
    with ProcessPoolExecutor(Processes) as ex:
        ex.map(backtrack_1_neutrino, y0_Nr)  


    print('Execution time:', time.time()-start, 'seconds.')