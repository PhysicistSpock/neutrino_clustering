from shared.preface import *
import shared.my_units as my
import shared.control_center as CC


#
### Functions used in simulation.
#

def rho_NFW(r, rho_0, r_s):
    """NFW density profile.

    Args:
        r (array): radius from center
        rho_0 (array): normalisation 
        r_s (array): scale radius

    Returns:
        array: density at radius r in [Msun/kpc**3]
    """    

    rho = rho_0 / (r/r_s) / np.power(1.+(r/r_s), 2.)

    return rho.to(unit.M_sun/unit.kpc**3.)


@nb.njit
def c_vir_avg(z, M_vir):
    # Functions from Mertsch et al. (2020), eqns. (12) and (13) in ref. [40].
    a_of_z = 0.537 + (0.488)*np.exp(-0.718*np.power(z, 1.08))
    b_of_z = -0.097 + 0.024*z

    # Argument in log has to be dimensionless
    arg_in_log = (M_vir / (1.e12 / my.h))

    # Calculate avergae c_vir
    c_vir_avg = np.power(a_of_z + b_of_z*np.log10(arg_in_log), 10.)

    return c_vir_avg
    

@nb.njit
def c_vir(z, M_vir):
    """Concentration parameter defined as r_vir/r_s, i.e. the ratio of virial 
    radius to the scale radius of the halo according to eqn. 5.5 of 
    Mertsch et al. (2020). 

    Args:
        z (array): redshift
        M_vir (float): virial mass, treated as fixed in time

    Returns:
        array: concentration parameters at each given redshift [dimensionless]
    """

    # Beta is then obtained from c_vir_avg(0, M_vir) and c_vir(0, M_vir).
    beta = (333.5/19.9) / c_vir_avg(0, M_vir)

    c = beta * c_vir_avg(z, M_vir)

    return c


@nb.njit
def rho_crit(z):
    """Critical density of the universe as a function of redshift, assuming
    matter domination, only Omega_m and Omega_Lambda in Friedmann equation. See 
    notes for derivation.

    Args:
        z (array): redshift

    Returns:
        array: critical density at redshift z [Msun/kpc**3]
    """    
    
    H_squared = my.H0**2. * (my.Omega_m0*(1.+z)**3. + my.Omega_L0) 
    rho_crit = 3.*H_squared / (8.*np.pi*const.G)

    # .to(unit.M_sun/unit.kpc**3.) invalid for numba, manual factor instead.
    return rho_crit / 64439975278.849915 


@nb.njit
def Omega_m(z):
    """Matter density parameter as a function of redshift, assuming matter
    domination, only Omega_m and Omega_Lambda in Friedmann equation. See notes
    for derivation.

    Args:
        z (array): redshift

    Returns:
        array: matter density parameter at redshift z [dimensionless]
    """    

    Omega_m = (my.Omega_m0*(1.+z)**3.)/(my.Omega_m0*(1.+z)**3.+my.Omega_L0)

    return Omega_m


@nb.njit
def Delta_vir(z):
    """Function as needed for their eqn. (5.7).

    Args:
        z (array): redshift

    Returns:
        array: value as specified just beneath eqn. (5.7) [dimensionless]
    """    

    Delta_vir = 18.*np.pi**2. + 82.*(Omega_m(z)-1.) - 39.*(Omega_m(z)-1.)**2.

    return Delta_vir


@nb.njit
def R_vir(z, M_vir):
    """Virial radius according to eqn. 5.7 in Mertsch et al. (2020).

    Args:
        z (array): redshift
        M_vir (float): virial mass

    Returns:
        array: virial radius [kpc]
    """    

    R_vir = np.power(3.*M_vir / (4.*np.pi*Delta_vir(z)*rho_crit(z)), 1./3.)

    return R_vir


@nb.njit
def scale_radius(z, M_vir):
    """Scale radius of NFW halo.

    Args:
        z (array): redshift
        M_vir (float): virial mass

    Returns:
        arrat: scale radius [kpc]
    """    
    
    r_s = R_vir(z, M_vir) / c_vir(z, M_vir)

    return r_s


#
### Utility functions.
#

def velocity_limits_of_m_nu(lower, upper, m_sim_eV, mode='kpc/s'):
    """Converts limits on p/T_nu to limits on velocity used in simulation."""

    # Conversion factor for limits from [eV] to [m/s] based on m_sim_eV
    m_sim_kg = m_sim_eV.to(unit.kg, unit.mass_energy())
    cf = my.T_nu_eV.to(unit.J) / m_sim_kg / const.c

    if mode == 'km/s':
        low_v = lower * cf.to(unit.km/unit.s)
        upp_v = upper * cf.to(unit.km/unit.s)
    if mode == 'kpc/s':
        low_v = lower * cf.to(my.Uunit)
        upp_v = upper * cf.to(my.Uunit)       

    return low_v, upp_v


def load_u_sim():
    # Load initial and final velocities of simulation.
    Ns = np.arange(CC.NR_OF_NEUTRINOS, dtype=int)  # Nr. of neutrinos
    sim = np.array([np.load(f'neutrino_vectors/nu_{Nr+1}.npy') for Nr in Ns])
    u_all = sim[:,:,3:6]  # (10000, 100, 3) shape, ndim = 3

    return u_all


def u_to_p_eV(u_sim, m_target_eV):
    """Converts velocities [kpc/s] (x,y,z from simulation) to 
    magnitude of momentum [eV] and ratio y=p/T_nu, according to desired
    target mass (and mass used in simulation)."""

    # Conversions
    u_sim_ms = (u_sim*unit.kpc/unit.s).to(unit.m/unit.s)

    # Magnitude of velocity
    if u_sim.ndim in (0,1):
        mag_sim = np.sqrt(np.sum(u_sim_ms**2))
    elif u_sim.ndim == 3:
        mag_sim = np.sqrt(np.sum(u_sim_ms**2, axis=2))
    else:
        mag_sim = np.sqrt(np.sum(u_sim_ms**2, axis=1))


    # From u_sim to p_sim; then from [Joule] to [eV].
    p_sim_eV = ((mag_sim * const.c * CC.NU_MASS_KG).to(unit.J)).to(unit.eV)
    
    # From p_sim to p_target.
    p_target_eV = p_sim_eV * (m_target_eV/CC.NU_MASS).value

    # p/T_nu ratio.
    y = (p_target_eV/my.T_nu_eV).value

    return p_target_eV, y


def y_fmt(value, tick_number):
    if value == 1e-2:
        return r'1+$10^{-2}$'
    elif value == 1e-1:
        return r'1+$10^{-1}$'
    elif value == 1e0:
        return r'1+$10^0$'
    elif value == 1e1:
        return r'1+$10^1$'


#
### Main functions.
#


def s_of_z(z):
    """Convert redshift to time variable s with eqn. 4.1 in Mertsch et al.
    (2020), keeping only Omega_m0 and Omega_Lambda0 in the Hubble eqn. for H(z).

    Args:
        z (float): redshift

    Returns:
        float: time variable s (in [seconds] if 1/H0 factor is included)
    """    

    def s_integrand(z):

        # original H0 in units ~[1/s], we only need the value
        H0 = my.H0.to(unit.s**-1.).value

        a_dot = np.sqrt(my.Omega_m0*(1.+z)**3. + my.Omega_L0)/(1.+z)*H0
        s_int = 1./a_dot

        return s_int

    s_of_z, _ = quad(s_integrand, 0., z)

    return np.float64(s_of_z)


@nb.njit
def grav_pot(x_i, z, rho_0, M_vir):

    # Compute values dependent on redshift.
    r_vir = R_vir(z, M_vir)
    r_s = r_vir / c_vir(z, M_vir)
    
    # Distance from halo center with current coords. x_i.
    r = np.sqrt(np.sum(x_i**2.))

    m = np.minimum(r, r_vir)
    M = np.maximum(r, r_vir)

    #! Ratios has to be unitless (e.g. else np.log yields 0.).
    ratio1 = (m/r_s)
    ratio2 = (r/r_s)
    ratio3 = (r_vir/M)
    ratio4 = (r_vir/r_s)

    # Gravitational potential in compact notation with m and M.
    prefactor = -4.*np.pi*const.G*rho_0*r_s**2.
    term1 = np.log(1.+ratio1) / ratio2
    term2 = ratio3 / (1.+ratio4)
    grav_pot = prefactor * (term1 - term2)

    # .to(unit.m**2/unit.s**2.) invalid for numba, manual factor instead.
    return grav_pot * 64439975278.84992


def escape_momentum(x_i, z, rho_0, M_vir, masses)

    # Gravitational potential at position x_i.
    grav = grav_pot(x_i, z, rho_0, M_vir)*unit.m**2/unit.s**2

    # Escape momentum formula from Ringwald & Wong (2004).
    p_esc = np.sqrt(2*np.abs(grav)) * masses.to(unit.kg, unit.mass_energy())
    p_esc_eV = ((p_esc * const.c).to(unit.J)).to(unit.eV)
    y_esc = (p_esc_eV/my.T_nu_eV).value

    return p_esc_eV, y_esc


@nb.njit
def dPsi_dxi_NFW(x_i, z, rho_0, M_vir):
    """Derivative of NFW grav. potential w.r.t. any axis x_i.

    Args:
        x_i (array): spatial position vector
        z (array): redshift
        rho_0 (float): normalization
        M_vir (float): virial mass

    Returns:
        array: Derivative vector of grav. potential. for all 3 spatial coords.
               with units of acceleration.
    """    

    # Compute values dependent on redshift.
    r_vir = R_vir(z, M_vir)
    r_s = r_vir / c_vir(z, M_vir)
    
    # Distance from halo center with current coords. x_i.
    r = np.sqrt(np.sum(x_i**2.))
    
    # Implement constant density core below certain radius.
    # R_core = r_vir
    # if r <= R_core:
    #     r = R_core

    m = np.minimum(r, r_vir)
    M = np.maximum(r, r_vir)

    #! Ratios has to be unitless (e.g. else np.log yields 0.).
    ratio1 = (m/r_s)
    ratio2 = (r/r_s)
    ratio3 = (r_vir/M)

    # Derivative in compact notation with m and M.
    #NOTE: Take absolute value of coord. x_i., s.t. derivative is never < 0.
    prefactor = 4.*np.pi*const.G*rho_0*r_s**2.*np.abs(x_i)/r**2.
    term1 = np.log(1.+ratio1) / ratio2
    term2 = ratio3 / (1.+ratio1)
    derivative_vector = prefactor * (term1 - term2)

    # .to(unit.kpc/unit.s**2.) invalid for numba, manual factor instead.
    return derivative_vector / 1.4775620405992722e28 


def Fermi_Dirac(p):
    """Fermi-Dirac phase-space distribution for CNB neutrinos. 
    Zero chem. potential and temp. T_nu (CNB temp. today). 

    Args:
        p (array): magnitude of momentum, must be in eV!

    Returns:
        array: Value of Fermi-Dirac distr. at p.
    """

    # Plug into Fermi-Dirac distribution 
    arg_of_exp = (p/(my.T_nu_eV)).value
    f_of_p = expit(-arg_of_exp)    

    return f_of_p


def number_density(p0, p1):
    """Neutrino number density obtained by integration over initial momenta.

    Args:
        p0 (array): neutrino momentum today
        p1 (array): neutrino momentum at z_back (final redshift in sim.)

    Returns:
        array: Value of relic neutrino number density.
    """    

    g = 2.  # 2 d.o.f.: flavour and anti-particle/particle 
    
    #NOTE: trapz integral method needs sorted (ascending) arrays
    ind = p0.argsort()
    p0_sort, p1_sort = p0[ind], p1[ind]

    # Fermi-Dirac value with momentum at end of sim.
    FDvals = Fermi_Dirac(p1_sort)  #! needs p in [eV]

    # Convert initial momentum p0 from [eV] to [kg*m/s]
    p0_SI = (p0_sort.to(unit.J) / const.c).to(unit.kg*unit.m/unit.s)
    
    # Calculate number density.
    y = p0_SI.value**2. * FDvals
    x = p0_SI.value
    n_raw = np.trapz(y, x)

    # Reintroduce "invisible" planck constant h to get 1/m**3
    #NOTE: leftover constants are g*4*np.pi
    n_m3 = n_raw / const.h.value**3 * g*4*np.pi

    # To 1/cm**3
    n_cm3 = (n_m3/unit.m**3).to(1/unit.cm**3)

    return n_cm3