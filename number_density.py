from shared.preface import *
import shared.my_units as my
import shared.functions as fct
import shared.control_center as CC


def number_density_1_mass(m_nu_eV):

    # Amount of simulated neutrinos
    Ns = np.arange(CC.NR_OF_NEUTRINOS, dtype=int)

    # load initial and final velocity
    u0 = [np.load(f'neutrino_vectors/nu_{Nr+1}.npy')[0][3:6] for Nr in Ns]
    u1 = [np.load(f'neutrino_vectors/nu_{Nr+1}.npy')[-1][3:6] for Nr in Ns]

    # intermediate arrays
    a0 = np.array([np.sqrt(np.sum(u**2)) for u in np.array(u0)])
    a1 = np.array([np.sqrt(np.sum(u**2)) for u in np.array(u1)])

    # convert mass(es) from eV to kg
    m_nu_kg = m_nu_eV.to(unit.kg, unit.mass_energy())

    n_nus = np.zeros(len(m_nu_kg))
    for i, m in enumerate(m_nu_kg.value):

        # convert velocities to momenta
        p0, p1 = a0 * m, a1 * m
        print(p0[0]-p1[0])

        #NOTE: number_density function need input momenta in units [kg*kpc/s]
        p_unit = unit.kg*unit.kpc/unit.s
        n_nus[i] = fct.number_density(p0*p_unit, p1*p_unit).value

    np.save('neutrino_data/number_densities.npy', n_nus)


if __name__ == '__main__':

    # 10 to 300 meV like in the paper
    mass_range_eV = np.linspace(0.01, 0.3, 2) * unit.eV

    number_density_1_mass(mass_range_eV)

    n_nus = np.load('neutrino_data/number_densities.npy')

    plt.loglog(mass_range_eV, n_nus)
    plt.xlabel('neutrino mass [eV]')
    plt.ylabel(r'number density [1/$cm^3$]')
    plt.show()