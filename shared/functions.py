from shared.preface import *
import shared.my_units as my

def rho_NFW(r, rho_0, r_s):
    """NFW density profile.

    Args:
        r (array): radius from center
        rho_0 (array): normalisation 
        r_s (array): scale radius

    Returns:
        array: density at radius r in [Msun/kpc**3]
    """    

    rho = rho_0 / (r/r_s) / np.power(1+(r/r_s), 2)

    return rho.to(unit.M_sun/unit.kpc**3)


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
    
    # Functions from Mertsch et al. (2020)
    a_of_z = 0.537 + (1.025-0.537)*np.exp(-0.718*np.power(z, 1.08))
    b_of_z = -0.097 + 0.025*z

    log10_beta_c = a_of_z + \
                   b_of_z*np.log10(M_vir / (1e12 * my.h**-1 * unit.M_sun))

    beta_c = np.power(log10_beta_c, 10)

    # beta factor calculated by using their values for scale and virial radius
    # in Table 1.
    beta = 333.5/19.9/beta_c

    c = beta_c*beta

    return c


def rho_crit(z):
    """Critical density of the universe as a function of redshift, assuming
    matter domination, only Omega_m and Omega_k in Friedmann equation. See 
    notes for derivation.

    Args:
        z (array): redshift

    Returns:
        array: critical density at redshift z [Msun/kpc**3]
    """    
    
    H = np.sqrt(1+my.Omega_m0*z) * (1+z) * my.H0
    rho_crit = 3*H**2 / (8*np.pi*const.G)

    return rho_crit.to(unit.M_sun/unit.kpc**3)


def Omega_m(z):
    """Matter density parameter as a function of redshift, assuming matter
    domination, only Omega_m and Omega_Lambda in Friedmann equation. See notes
    for derivation.

    Args:
        z (array): redshift

    Returns:
        array: matter density parameter at redshift z [dimensionless]
    """    

    return np.float64((my.Omega_m0*(1+z)**3) / (1 + my.Omega_m0*((1+z)**3-1)))


def Delta_vir(z):
    """Function as needed for their eqn. (5.7).

    Args:
        z (array): redshift

    Returns:
        array: value as specified just beneath eqn. (5.7) [dimensionless]
    """    

    Delta_vir = 18*np.pi**2 + 82*(Omega_m(z)-1) - 39*(Omega_m(z)-1)**2

    return Delta_vir


def R_vir(z, M_vir):
    """Virial radius according to eqn. 5.7 in Mertsch et al. (2020).

    Args:
        z (array): redshift
        M_vir (float): virial mass

    Returns:
        array: virial radius [kpc]
    """    

    R_vir = np.power(3*M_vir / (4*np.pi*Delta_vir(z)*rho_crit(z)), 1/3)

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


def s_of_z(z):
    """Convert redshift to time variable s with eqn. 4.1 in Mertsch et al.
    (2020), keeping only Omega_m0 and Omega_Lambda0 in the Hubble eqn. for H(z).

    Args:
        z (float): redshift

    Returns:
        float: time variable s (in [seconds] due to 1/H0 factor)
    """    

    def s_integrand(z):

        # original H0 in units ~[1/s], we only need the value
        H0 = my.H0.to(unit.s**-1).value

        s_int = -1/H0/np.sqrt((my.Omega_m0*(1+z)**3 + my.Omega_L0))

        return s_int

    s_of_z, _ = quad(s_integrand, 0, z)

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
               with units of acceleration
    """    

    # compute values dependent on redshift
    r_vir = R_vir(z, M_vir)
    r_s = r_vir / c_vir(z, M_vir)
    
    # distance from halo center with current coords. x_i
    r0 = np.sqrt(np.sum(x_i**2))

    ### This is for the whole expression as in eqn. (A.5) and using sympy
    # region

    # m = np.minimum(r0, r_vir)
    # M = np.maximum(r0, r_vir)

    # r = sympy.Symbol('r')

    # prefactor = -4*np.pi*unit.G*rho_0*r_s**2
    # term1 = np.log(1 + m/r_s) / (r/r_s)
    # term2 = r_vir/M / (1 + r_vir/r_s)
    # Psi = prefactor * (term1 - term2)

    ## derivative w.r.t any axis x_i with chain rule
    # dPsi_dxi = sympy.diff(Psi, r) * x_i / r0

    ## fill in r values
    # fill_in_r = sympy.lambdify(r, dPsi_dxi, 'numpy')
    # derivative_vector = fill_in_r(r0)

    # endregion

    m = np.minimum(r0, r_vir)

    # ratio has to be unitless, otherwise np.log yields 0.
    ratio = m.value/r_s.value

    prefactor = -4*np.pi*const.G*rho_0*r_s**2 * np.log(1+(ratio)) * r_s
    derivative = prefactor / r0**2
    derivative_vector = derivative * x_i/r0

    return derivative_vector.to(unit.kpc/unit.s**2)


def Fermi_Dirac(p, m=0.):
    """Fermi-Dirac phase-space distribution for neutrinos with 
    zero chem. potential and temp. T_nu (CNB temp. today).

    Args:
        p (array): magnitude of momentum
        m (float): mass of particle species

    Returns:
        array: Value of Fermi-Dirac distr. at p.
    """    

    # with numpy, trouble with overflow
    #TODO: ratio in exponent has to be dimensionless, introduce const.k_B
    # multiplying my.T_nu, introduce c's in p**2+m**2
    f_of_p = 1 / (np.exp(np.sqrt(p**2+m**2)/my.T_nu) + 1)

    # with scipy
    # f_of_p = expit(-np.sqrt(p**2+m**2)/unit.T_nu)

    return f_of_p


def number_density(p0, p_back):
    """Neutrino number density obtained by integration over initial momenta.

    Args:
        p0 (array): neutrino momentum today
        p_back (array): neutrino momentum at z_back (final redshift in sim.) 

    Returns:
        array: Value of relic neutrino number density.
    """    

    g = 1  # 6 degrees of freedom: flavour and particle/anti-particle ?

    #TODO: there are hidden units/constants here, ultimately we want /cm**3.
    n = 1/(2*np.pi**2) * np.sum(p0**2 * Fermi_Dirac(p_back))

    return n / unit.cm**3