from shared.preface import *
import shared.my_units as my
import shared.functions as fct
import shared.control_center as CC



def bin_volumes(radial_bins):
    """Returns the volumes of the bins. """

    single_vol = lambda x: (4.0 / 3.0) * np.pi * x ** 3
    outer = single_vol(radial_bins[1:])
    inner = single_vol(radial_bins[:-1])
    return outer - inner


def bin_centers(radial_bins):
    """Returns the centers of the bins. """

    outer = radial_bins[1:]
    inner = radial_bins[:-1]
    return 0.5 * (outer + inner)


def analyse_halo(mass, pos):
    # Define radial bins [log scale, kpc units]
    radial_bins = np.arange(0, 5, 0.1)
    radial_bins = 10 ** radial_bins

    # Radial coordinates [kpc units]
    r = np.sqrt(np.sum(pos ** 2, axis=1))

    SumMasses, _, _ = stat.binned_statistic(x=r, values=np.ones(len(r)) * mass[0], statistic="sum", bins=radial_bins, )
    density = (SumMasses / bin_volumes(radial_bins))  # Msun/kpc^3
    return density


def read_data(which_halos,snap,folder,output_path,name,mass_select):

    radial_bins = np.arange(0, 5, 0.1)
    radial_bins = 10**radial_bins
    centers = bin_centers(radial_bins) #kpc

    with h5py.File(folder+"/snapshot_00%02i.hdf5"%snap) as hf:
        a = hf["/Header"].attrs["Scale-factor"]
        mass = hf['PartType1/Masses'][:] * 1e10 #Msun
        pos = hf['PartType1/Coordinates'][:][:] * a
        vel = hf['PartType1/Velocities'][:][:]
        unit_length_in_cgs = hf["/Units"].attrs["Unit length in cgs (U_L)"]

    snaps = h5py.File('/home/fabian/ownCloud/sim_data/snapshot_0036.hdf5')
    group = h5py.File('/home/fabian/ownCloud/sim_data/subhalo_0036.catalog_groups')
    parts = h5py.File('/home/fabian/ownCloud/sim_data/subhalo_0036.catalog_particles')
    props = h5py.File('/home/fabian/ownCloud/sim_data/subhalo_0036.properties')

    m200c = props["Mass_200crit"][:] * 1e10
    m200c[m200c == 0] = 1
    m200c = np.log10(m200c)
    CoP = np.zeros((len(m200c), 3))
    CoP[:, 0] = props["Xcminpot"][:]
    CoP[:, 1] = props["Ycminpot"][:]
    CoP[:, 2] = props["Zcminpot"][:]
    subtype = props["Structuretype"][:]

    if mass_select == 10:
        select_halos = np.where((m200c >= 9.8) & (m200c <= 10.2))[0]  # >10 star parts
    if mass_select == 11:
        select_halos = np.where((m200c >= 10.8) & (m200c <= 11.2))[0]  # >10 star parts
    if mass_select == 12:
        select_halos = np.where((m200c >= 11.8) & (m200c <= 12.2))[0]  # >10 star parts

    # Checking sample
    if which_halos == 'subhalos':
        select = np.where(subtype[select_halos] > 10)[0]
        select_halos = select_halos[select]
    else:
        select = np.where(subtype[select_halos] == 10)[0]
        select_halos = select_halos[select]

    if len(select_halos) >= 30:
        select_random = np.random.random_integers(len(select_halos) - 1, size=(30))
        select_halos = select_halos[select_random]


    M200 = np.median(10 ** m200c[select_halos])
    num_halos = len(select_halos)

    density_all = np.zeros((len(centers), num_halos))

    for halo in range(0, num_halos):
        halo_j = select_halos[halo]

        # Grab the start position in the particles file to read from
        halo_start_position = group["Offset"][halo_j]
        halo_end_position = group["Offset"][halo_j + 1]
        particle_ids_in_halo = parts["Particle_IDs"][halo_start_position:halo_end_position]
        particle_ids_from_snapshot = snaps["PartType1/ParticleIDs"][...]

        _, indices_v, indices_p = np.intersect1d(particle_ids_in_halo,
                                                 particle_ids_from_snapshot,
                                                 assume_unique=True,
                                                 return_indices=True, )

        particles_mass = mass[indices_p].copy()
        particles_pos = pos[indices_p, :].copy()
        particles_pos -= CoP[halo_j, :]  # centering
        particles_pos *= 1e3  # kpc
        if len(particles_mass) == 0 :continue
        density_all[:, halo] = analyse_halo(particles_mass, particles_pos)

    densityM = np.median(density_all[:, :], axis=1)
    densityUp = np.percentile(density_all[:, :], 84, axis=1)
    densityLow = np.percentile(density_all[:, :], 16, axis=1)

    # Output final median profile:
    output = np.zeros((len(centers),4))
    output[:,0] = centers
    output[:,1] = densityM
    output[:,2] = densityLow
    output[:,3] = densityUp

    if mass_select == 10:
        if which_halos == 'subhalos':
            np.savetxt(f'sim_data/Profile_subhalos_M{mass_select}_{name}.txt')
        else:
            np.savetxt(f'sim_data/Profile_halos_M{mass_select}_{name}.txt')
    if mass_select == 11:
        if which_halos == 'subhalos':
            np.savetxt(f'sim_data/Profile_subhalos_M{mass_select}_{name}.txt')
        else:
            np.savetxt(f'sim_data/Profile_halos_M{mass_select}_{name}.txt')
    if mass_select == 12:
        if which_halos == 'subhalos':
            np.savetxt(f'sim_data/Profile_subhalos_M{mass_select}_{name}.txt')
        else:
            np.savetxt(f'sim_data/Profile_halos_M{mass_select}_{name}.txt')
