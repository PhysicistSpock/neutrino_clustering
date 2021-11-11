from shared.preface import *
import shared.my_units as my
import shared.functions as fct
import shared.control_center as CC


def draw_ui(phi_points, theta_points, v_points):
    """Get initial velocities for the neutrinos."""
    
    # Conversion factor for limits.
    cf = my.T_nu_eV.to(unit.J) / CC.NU_MASS_KG / const.c

    # Limits on velocity.
    lower = CC.LOWER * cf.to(my.Uunit)
    upper = CC.UPPER * cf.to(my.Uunit)

    # Initial magnitudes of the velocities.
    v_kpc = np.geomspace(lower.value, upper.value, v_points)

    # Split up this magnitude into velocity components.
    #NOTE: Fone by using spher. coords. trafos, which act as "weights".

    eps = 0.01  # shift in theta, so poles are not included
    ps = np.linspace(0., 2.*np.pi, phi_points)
    ts = np.linspace(0.+eps, np.pi-eps, theta_points)

    # Minus sign for all due to choice of coord. system setup (see drawings).
    uxs = [-v*np.cos(p)*np.sin(t) for p in ps for t in ts for v in v_kpc]
    uys = [-v*np.sin(p)*np.sin(t) for p in ps for t in ts for v in v_kpc]
    uzs = [-v*np.cos(t) for _ in ps for t in ts for v in v_kpc]

    ui_array = np.array([[ux, uy, uz] for ux,uy,uz in zip(uxs,uys,uzs)])        

    return ui_array 


def EOMs(s, y):
    """Equations of motion for all x_i's and u_i's in terms of s."""

    # Initialize vector and attach astropy units.
    x_i_vals, u_i_vals = np.reshape(y, (2,3))
    x_i, u_i = x_i_vals*my.Xunit, u_i_vals*my.Uunit

    # Find z corresponding to s.
    if s in s_steps:
        z = CC.ZEDS[s_steps==s][0]
    else:
        z = s_to_z(s)  # interpolation function defined below

    # Gradient value will always be positive.
    gradient = fct.dPsi_dxi_NFW(x_i, z, my.rho0_NFW, my.Mvir_NFW).value

    #NOTE: Velocity has to change according to the pointing direction,
    #NOTE: treat all 4 cases seperately.
    signs = np.zeros(3)
    for i, (pos, vel) in enumerate(zip(x_i, u_i)):
        if pos > 0. and vel > 0.:
            signs[i] = -1.
        elif pos > 0. and vel < 0.:
            signs[i] = -1.
        elif pos < 0. and vel > 0.:
            signs[i] = +1.
        else:  # pos < 0. and vel < 0.
            signs[i] = +1.
    
    # Create dx/ds and du/ds, i.e. the r.h.s of the eqns. of motion. 
    u_i_kpc = u_i.to(my.Uunit).value

    dyds = CC.TIME_FLOW * np.array([
        u_i_kpc, signs * 1./(1.+z)**2. * gradient
    ])

    return dyds


def backtrack_1_neutrino(y0_Nr):
    """Simulate trajectory of 1 neutrino."""

    # Split input into initial vector and neutrino number.
    y0, Nr = y0_Nr[0:-1], y0_Nr[-1]

    # Solve all 6 EOMs.
    sol = solve_ivp(
        fun=EOMs, t_span=[s_steps[0], s_steps[-1]], t_eval=s_steps,
        y0=y0, method=CC.SOLVER, vectorized=True
        )
    
    #NOTE: output as raw numbers but in [kpc, kpc/s]
    np.save(f'neutrino_vectors/nu_{int(Nr)}.npy', np.array(sol.y.T))


if __name__ == '__main__':
    start = time.time()

    # Integration steps.
    s_steps = np.array([fct.s_of_z(z) for z in CC.ZEDS])
    s_to_z = interp1d(s_steps, CC.ZEDS, kind='linear', fill_value='extrapolate')

    # Amount of neutrinos to simulate.
    nu_Nr = CC.NR_OF_NEUTRINOS

    # Position of earth w.r.t Milky Way NFW halo center.
    #NOTE: Earth is placed on x axis of coord. system.
    x1, x2, x3 = 8.5, 0., 0.
    x0 = np.array([x1, x2, x3])

    # Draw initial velocities.
    #NOTE: in kpc/s (without astropy unit attached)
    ui = draw_ui(
        phi_points   = CC.PHIs,
        theta_points = CC.THETAs,
        v_points     = CC.Vs
        )
    
    # Combine vectors and append neutrino particle number.
    y0_Nr = np.array([np.concatenate((x0,ui[i],[i+1])) for i in range(nu_Nr)])

    # Run simulation on multiple cores.
    Processes = 32
    with ProcessPoolExecutor(Processes) as ex:
        ex.map(backtrack_1_neutrino, y0_Nr)  

    seconds = time.time()-start
    minutes = seconds/60.
    hours = minutes/60.
    print('Time sec/min/h: ', seconds, minutes, hours)