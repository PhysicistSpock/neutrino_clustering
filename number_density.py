from shared.preface import *
import shared.my_units as my
import shared.functions as fct
import shared.control_center as CC


if __name__ == '__main__':

    neutrinos = CC.NR_OF_NEUTRINOS

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
    print('Final number density:', n_nu)