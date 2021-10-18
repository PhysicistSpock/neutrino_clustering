from shared.preface import *
import shared.my_units as my
import shared.control_center as CC


#
### Utility functions.
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
    
    H_squared = my.H0**2 * (my.Omega_m0*(1+z)**3 + my.Omega_L0) 
    rho_crit = 3*H_squared / (8*np.pi*const.G)

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
        H0 = my.H0.to(unit.s**-1).value

        a_dot = np.sqrt(my.Omega_m0*(1+z)**3 + my.Omega_L0)*H0
        s_int = 1/a_dot

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
               with units of acceleration.
    """    

    # compute values dependent on redshift
    r_vir = R_vir(z, M_vir)
    r_s = r_vir / c_vir(z, M_vir)
    
    # distance from halo center with current coords. x_i
    r = np.sqrt(np.sum(x_i**2))
    if r == 0.:
        r = 0.01  # avoid singularity

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

    m = np.minimum(r, r_vir)

    #NOTE ratio has to be unitless, otherwise np.log yields 0.
    ratio = m.value/r_s.value

    prefactor = -4*np.pi*const.G*rho_0*r_s**2 * np.log(1+(ratio)) * r_s
    derivative = (-1) * prefactor / r**2

    #! Take absolute value of position coord. x_i, we want only the magnitude
    #! of the derivative. 
    derivative_vector = derivative * np.abs(x_i)/r

    return derivative_vector.to(unit.kpc/unit.s**2)


def Fermi_Dirac(p, z_back):
    """Fermi-Dirac phase-space distribution for CNB neutrinos. 
    Zero chem. potential and temp. T_nu (CNB temp. today). 

    Args:
        p (array): magnitude of momentum, must be in eV!

    Returns:
        array: Value of Fermi-Dirac distr. at p.
    """

    
    #NOTE: Temp. at z_back is higher than T_nu today.
    T_zback_eV = my.T_nu_eV*(1+z_back)
    # T_zback_eV = my.T_nu_eV*(1)  # Today, used for testing Fermi-Dirac shape

    # Plug into Fermi-Dirac distribution 
    arg_of_exp = p/T_zback_eV
    f_of_p = 1 / (np.exp(arg_of_exp.value) + 1)

    return f_of_p


def number_density(p0, p_back, z_back):
    """Neutrino number density obtained by integration over initial momenta.

    Args:
        p0 (array): neutrino momentum today
        p_back (array): neutrino momentum at z_back (final redshift in sim.)

    Returns:
        array: Value of relic neutrino number density.
    """    

    g = 1  #? 6 degrees of freedom: flavour and particle/anti-particle

    # convert momenta from kg*kpc/s to eV
    to_eV = 1/(5.344286e-28)
    p0 = p0.to(unit.kg*unit.m/unit.s).value * to_eV
    p_back = p_back.to(unit.kg*unit.m/unit.s).value * to_eV

    #NOTE: trapz integral method needs sorted (ascending) arrays
    order = p0.argsort()
    p0_sort, p_back_sort = p0[order]*unit.eV, p_back[order]*unit.eV
    #! Fermi_Dirac function needs p to have units of eV attached

    # precomputed factors
    prefactor = g/(2*np.pi**2)
    FDvals = Fermi_Dirac(p_back_sort, z_back)

    #NOTE: n ~ integral dp p**2 f(p), the units come from dp p**2, which have
    #NOTE: eV*3 = 1/eV**-3 ~ 1/length**3
    n = prefactor * np.trapz(p0_sort.value**2 * FDvals, p0_sort.value)

    # convert n from eV**3 (also by hc actually) to 1/cm**3
    ev_by_hc_to_cm_neg1 = (1/const.h/const.c).to(1/unit.cm/unit.eV)
    n_cm3 = n * ev_by_hc_to_cm_neg1.value**3 / unit.cm**3

    return n_cm3


def number_density_eV(p0, p_back, z_back):
    """Neutrino number density obtained by integration over initial momenta.

    Args:
        p0 (array): neutrino momentum today
        p_back (array): neutrino momentum at z_back (final redshift in sim.)

    Returns:
        array: Value of relic neutrino number density.
    """    

    g = 1  #? 6 degrees of freedom: flavour and particle/anti-particle

    #NOTE: trapz integral method needs sorted (ascending) arrays
    order = p0.value.argsort()
    p0_sort, p_back_sort = p0[order], p_back[order]

    # precomputed factors
    const = g/(2*np.pi**2)
    FDvals = Fermi_Dirac(p_back_sort, z_back)

    #NOTE: n ~ integral dp p**2 f(p), the units come from dp p**2, which have
    #NOTE: eV*3 = 1/eV**-3 ~ 1/length**3
    n = const * np.trapz(p0_sort.value**2 * FDvals, p0_sort.value)

    # convert n from eV**3 to 1/cm**3
    eV_to_mneg1 = (1/1.97327e-7)
    n_m3 = n * eV_to_mneg1**3 * (1/unit.m**3)
    n_cm3 = n_m3.to(1/unit.cm**3)

    return n_cm3