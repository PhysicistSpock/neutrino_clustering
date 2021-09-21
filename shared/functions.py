from shared.preface import *
import shared.my_units as unit

def rho_NFW(r, rho_0, r_s):
    """NFW density profile.

    Args:
        r (array): radius from center
        rho_0 (array): normalisation 
        r_s (array): scale radius

    Returns:
        array: density at radius r [Msun/kpc**3]
    """    

    rho = rho_0 / (r/r_s) / np.power(1+(r/r_s), 2)

    return rho / (unit.Msun/unit.kpc**3)


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

    log10_c = a_of_z + \
              b_of_z*np.log10(M_vir / (10**12 * unit.h**-1 * unit.Msun))
    c = np.power(log10_c, 10)

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
    
    H = np.sqrt(1+unit.Omega_m0*z) * (1+z) * unit.H0
    rho_crit = 3*H**2 / (8*np.pi*unit.G)

    return rho_crit / (unit.Msun/unit.kpc**3)


def Omega_m(z):
    """Matter density parameter as a function of redshift, assuming matter
    domination, only Omega_m and Omega_k in Friedmann equation. See notes
    for derivation.

    Args:
        z (array): redshift

    Returns:
        array: matter density parameter at redshift z [dimensionless]
    """    

    o = unit.Omega_m0
    Omega_m = o*(1+z) / (1+o*z)

    return Omega_m


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

    return R_vir / unit.kpc


def scale_radius(z, M_vir):
    """Scale radius of NFW halo.

    Args:
        z (array): redshift
        M_vir (float): virial mass

    Returns:
        arrat: scale radius [kpc]
    """    
    
    r_s = R_vir(z, M_vir) / c_vir(z, M_vir)

    return r_s / unit.kpc


def s_of_z(z):
    """Convert redshift to time variable s with eqn. 4.1 in Mertsch et al.
    (2020), assuming matter domination.

    Args:
        z (array): redshift

    Returns:
        array: time variable s (in [seconds] due to 1/H0 factor)
    """    

    s = 2/unit.H0/unit.Omega_m0 * (1-np.sqrt(1+unit.Omega_m0*z))

    return s / unit.s


def z_of_s(s):
    """Convert time variable s to redshift with eqn 4.1 in Mertsch et al. 
    (2020), assuming matter domination.

    Args:
        s (array): time variable s as in eqn 4.1

    Returns:
        array: redshift [dimensionless]
    """    

    z = 1/unit.Omega_m0 * \
        ((1-(unit.H0*unit.Omega_m0/2*s))**2 - 1)

    return z


def dPsi_dxi_NFW(x_i, z, rho_0, M_vir):
    """Derivative of grav. potential w.r.t. any axis x_i.

    Args:
        x_i (array): spatial position vector
        z (array): redshift
        rho_0 (float): normalization
        M_vir (float): virial mass

    Returns:
        array: Derivative vector of grav. potential. for all 3 spatial coords.
    """    

    # compute values dependent on redshift
    r_vir = R_vir(z, M_vir)
    r_s = r_vir / c_vir(z, M_vir)
    
    # distance from halo center with current coords. x_i
    r0 = np.sqrt(np.sum(x_i**2))

    m = np.minimum(r0, r_vir)
    # M = np.maximum(r0, r_vir)

    r = sympy.Symbol('r')

    prefactor = -4*np.pi*unit.G*rho_0*r_s**2
    term1 = np.log(1 + m/r_s) / (r/r_s)
    # term2 = r_vir/M / (1 + r_vir/r_s)

    # term2 drops anyway when deriving w.r.t. r
    Psi = prefactor * (term1)# - term2)

    # derivative w.r.t any axis x_i with chain rule
    dPsi_dxi = sympy.diff(Psi, r) * x_i / r0

    # fill in r values
    fill_in_r = sympy.lambdify(r, dPsi_dxi, 'numpy')
    derivative_vector = fill_in_r(r0)
    
    # print('derivative values', derivative_vector)

    return np.array(derivative_vector)


def EOMs(y, s, rho_0, M_vir):

    # initialize vector
    x_i, u_i = np.reshape(y, (2,3))

    # convert time variable s to redshift
    z = z_of_s(s)

    # derivative of grav. potential
    derivative_vector = dPsi_dxi_NFW(x_i, z, rho_0, M_vir)

    # global minus sign for dydt array, s.t. calculation is backwards in time
    dydt = -np.array([u_i, -(1+z)**-2 * derivative_vector])

    # reshape from (2,3) to (6,), s.t. the vector is
    # (x_1, x_2, x_3, u_1, u_2, u_3)
    return np.reshape(dydt, 6)


def Fermi_Dirac(p):

    f_of_p = 1 / (np.exp(p/unit.T_nu) + 1)

    return f_of_p


def number_density(p):

    n = np.sum(p * Fermi_Dirac(p))

    return n