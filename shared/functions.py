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
    
    def c_vir_avg(z, M_vir):
        # Functions from Mertsch et al. (2020), eqns. (12) and (13) in ref. [40].
        a_of_z = 0.537 + (0.488)*np.exp(-0.718*np.power(z, 1.08))
        b_of_z = -0.097 + 0.024*z

        # Argument in log has to be dimensionless
        arg_in_log = (M_vir / (1.e12 / my.h * unit.M_sun)).value

        # Calculate avergae c_vir
        c_vir_avg = np.power(a_of_z + b_of_z*np.log10(arg_in_log), 10.)

        return c_vir_avg

    # Beta is then obtained from c_vir_avg(0, M_vir) and c_vir(0, M_vir).
    beta = (333.5/19.9) / c_vir_avg(0, M_vir)
    # beta = 2.09
    print(beta)

    c = beta * c_vir_avg(z, M_vir)

    return c


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

    return rho_crit.to(unit.M_sun/unit.kpc**3.)


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

    return np.float64(Omega_m)


def Delta_vir(z):
    """Function as needed for their eqn. (5.7).

    Args:
        z (array): redshift

    Returns:
        array: value as specified just beneath eqn. (5.7) [dimensionless]
    """    

    Delta_vir = 18.*np.pi**2. + 82.*(Omega_m(z)-1.) - 39.*(Omega_m(z)-1.)**2.

    return Delta_vir


def R_vir(z, M_vir):
    """Virial radius according to eqn. 5.7 in Mertsch et al. (2020).

    Args:
        z (array): redshift
        M_vir (float): virial mass

    Returns:
        array: virial radius [kpc]
    """    

    R_vir = np.power(3.*M_vir / (4.*np.pi*Delta_vir(z)*rho_crit(z)), 1./3.)

    return R_vir.to(unit.kpc)


def scale_radius(z, M_vir):
    """Scale radius of NFW halo.

    Args:
        z (array): redshift
        M_vir (float): virial mass

    Returns:
        arrat: scale radius [kpc]
    """    
    
    r_s = R_vir(z, M_vir) / c_vir(z, M_vir)

    return r_s.to(unit.kpc)


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


def u_to_p_eV(u_sim, m_sim_eV, m_target_eV):
    """Converts velocity [kpc/s] (from simulation) to momentum [eV]
    and ratio y=p/T_nu."""

    # Conversions
    m_sim_kg = m_sim_eV.to(unit.kg, unit.mass_energy())
    u_sim_ms = (u_sim*unit.kpc/unit.s).to(unit.m/unit.s)

    # Magnitude of velocity
    if u_sim.ndim in (0,1):
        mag_sim = np.sqrt(np.sum(u_sim_ms**2))
    else:
        mag_sim = np.sqrt(np.sum(u_sim_ms**2, axis=1))


    # From u_sim to p_sim, [Joule] then [eV]
    p_sim_eV = ((mag_sim * const.c * m_sim_kg).to(unit.J)).to(unit.eV)
    
    # From p_sim to p_target
    p_target_eV = p_sim_eV * (m_target_eV/m_sim_eV).value

    # p/T_nu ratio
    y = p_target_eV / my.T_nu_eV

    return p_target_eV, y


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
    if r == 0.:
        r = 1e-10  # avoid singularity

    m = np.minimum(r, r_vir)

    #! Ratio has to be unitless, otherwise np.log yields 0.
    ratio = (m/r_s).value

    prefactor = -4.*np.pi*const.G*rho_0*r_s**2.*np.log(1.+(ratio))*r_s
    derivative = (-1.) * prefactor / r**2.

    # Absolute value of strength of derivative at coord. x_i.
    derivative_vector = derivative * np.abs(x_i)/r

    return derivative_vector.to(unit.kpc/unit.s**2.)


def Fermi_Dirac(p, z):
    """Fermi-Dirac phase-space distribution for CNB neutrinos. 
    Zero chem. potential and temp. T_nu (CNB temp. today). 

    Args:
        p (array): magnitude of momentum, must be in eV!

    Returns:
        array: Value of Fermi-Dirac distr. at p.
    """

    # Plug into Fermi-Dirac distribution 
    arg_of_exp = p/my.T_nu_eV*(1.+z)
    f_of_p = 1. / (np.exp(arg_of_exp.value) + 1.)

    return f_of_p


def number_density(p0, p1, z):
    """Neutrino number density obtained by integration over initial momenta.

    Args:
        p0 (array): neutrino momentum today
        p1 (array): neutrino momentum at z_back (final redshift in sim.)

    Returns:
        array: Value of relic neutrino number density.
    """    

    g = 1.  #? 6 degrees of freedom: flavour and particle/anti-particle
    
    #NOTE: trapz integral method needs sorted (ascending) arrays
    order = p0.argsort()
    p0_sort, p1_sort = p0[order], p1[order]

    # precomputed factors
    prefactor = g/(2.*np.pi**2.)
    FDvals = Fermi_Dirac(p1_sort, z)  #! needs p in [eV]

    #NOTE: n ~ integral dp p**2 f(p), the units come from dp p**2, which have
    #NOTE: eV*3 = 1/eV**-3 ~ 1/length**3
    y = p0_sort.value**2. * FDvals
    x = p0_sort.value
    n = prefactor * np.trapz(y, x)

    # convert n from eV**3 (also by hc actually) to 1/cm**3
    ev_by_hc_to_cm_neg1 = (1./const.h/const.c).to(1./unit.cm/unit.eV)
    n_cm3 = n * ev_by_hc_to_cm_neg1.value**3. / unit.cm**3.

    return n_cm3


def Fermi_Dirac_cart(px, py, pz):
    
    # Plug into Fermi-Dirac distribution 
    arg_of_exp = np.sqrt((px**2+py**2+pz**2))/my.T_nu_eV
    f_of_p = 1. / (np.exp(arg_of_exp.value) + 1.)

    return f_of_p


def number_density_cart(p0x, p0y, p0z, p1x, p1y, p1z):
    
    g = 1.  #? 6 degrees of freedom: flavour and particle/anti-particle
    
    #NOTE: trapz integral method needs sorted (ascending) arrays
    o = p0x.argsort()
    p0x, p0y, p0z = p0x[o], p0y[o], p0z[o]
    p1x, p1y, p1z = p1x[o], p1y[o], p1z[o]

    # precomputed factors
    prefactor = g/(2.*np.pi**2.)
    FDvals = Fermi_Dirac(p1x, p1y, p1z)  #! needs p in [eV]

    # Fermi_Dirac function to integrate
    

    #NOTE: n ~ integral dp p**2 f(p), the units come from dp p**2, which have
    #NOTE: eV*3 = 1/eV**-3 ~ 1/length**3
    z_part = np.trapz(
        1./(np.exp((np.sqrt(p0x**2+p0y**2+p0z**2)/my.T_nu_eV).value)+1.), 
        p0x.value
        )


    # convert n from eV**3 (also by hc actually) to 1/cm**3
    ev_by_hc_to_cm_neg1 = (1./const.h/const.c).to(1./unit.cm/unit.eV)
    n_cm3 = n * ev_by_hc_to_cm_neg1.value**3. / unit.cm**3.

    return n_cm3