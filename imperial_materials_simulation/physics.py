'''
Vectorised functions and classes for efficiently calculating kintetic energy, temperature, spring bonding
interactions, and Lennard Jones long range interactions.
'''
import numpy as np
#compiles functions for much higher speed when called many times. Limited compatibility with numpy
import numba as nb 

def get_kintetic_energy(velocities, mass) -> np.float64:
    '''returns total kinetic energy of all atoms'''
    return np.sum(mass/2 * velocities**2)

def get_temperature(kinetic_energy: np.float64, n_atoms: int) -> np.float64:
    '''returns the average temperature of all the atoms'''
    kb = 8.617333262e-05
    return 2 * kinetic_energy / (3 * (n_atoms-1) * kb)

@nb.jit(nopython=True, fastmath=True) #fastmath increases float64 innaccuracy by 4x (1.1e-16 to 4.4e-16) which is acceptable here
def get_bonding_interactions(positions: np.ndarray, equilibrium_bond_length: float, spring_constant: float) \
                                -> list[np.ndarray, float]:
    '''returns bonding forces for each atom and the total bonding potential of all atoms'''
    bond_displacements = positions[1:] - positions[:-1]
    bond_lengths = np.sum(bond_displacements**2, axis=1) ** 0.5
    bond_lengths = bond_lengths.reshape(bond_lengths.shape[0], 1) #enables broadcasting in bond direction calculation
    bond_extensions = bond_lengths - equilibrium_bond_length
    bond_directions = bond_displacements / bond_lengths

    bond_forces = -spring_constant * bond_extensions * bond_directions
    atom_forces = np.zeros(shape=positions.shape)
    atom_forces[:-1] -= bond_forces #each bond exerts a negative force on the left atom
    atom_forces[1:] += bond_forces #each bond exerts a positive force on the right atom
    total_bond_potential = np.sum(spring_constant/2 * (bond_extensions**2))
    return atom_forces, total_bond_potential

#numpy broadcasting can be used to replace the for loop and increase speed. This however requires boolean indexing
#which is not supported by numba and so the proper numpy implementation actually ends up being slower.
@nb.jit(nopython=True, fastmath=True)
def get_non_bonding_interactions(positions: np.ndarray, epsilon: float, sigma: float) -> list[np.ndarray, float]:
    '''
    returns the Lennard Jones (LJ) force for each atom and the total LJ potential of all atoms. This approximates
    long range Van Der Valls attraction as well as the hard sphere repulsion of each atom to its more distant neighbours
    (equilibrium LJ seperation is ~3x the equilibrium bond seperation) 
    '''
    n_atoms = positions.shape[0]
    forces = np.zeros((n_atoms,3))
    potential = 0.0
    for i in range(n_atoms-1):
        #displacements from atom i to its 3rd nearest neighbour and beyond (LJ models long range interactions). Looks
        #like it ignores rightmost atoms but only find the displacement between relevant pairs once for efficiency 
        displacements = positions[i+3:] - positions[i]
        lengths = np.sum(displacements**2, axis=1) ** 0.5
        lengths = lengths.reshape(lengths.shape[0], 1) #allows broadcasting in following calculations
        directions = displacements / lengths
        sixpowers = (sigma/lengths) ** 6
        twelvepowers = sixpowers ** 2
        potential += np.sum(epsilon * (twelvepowers - 2*sixpowers))

        #forces exerted by atom i on its relevant neighbours
        atom_forces = epsilon * 12/lengths * (twelvepowers - sixpowers) * directions
        forces[i+3:] += atom_forces 
        forces[i] -= np.sum(atom_forces, axis=0) #equal and opposite reaction on atom

    return forces, potential 

@nb.jit(nopython=True, fastmath=True)
def get_energy_change(positions: np.ndarray, displacements: np.ndarray, lengths: np.ndarray, atom_index: int,
                      displacement: np.ndarray, epsilon: float, sigma: float, equilibrium_bond_length: float,
                      spring_constant: float) -> list[float, np.ndarray, np.ndarray, np.ndarray]:
    '''
    calculates only affected atom seperations to find how total potential energy changes. Really requires
    you to draw out a mock displacement matrix and pick an atom index to understand the array indexing.
    returns energy_change, new_positions, new_displacements, new_lengths
    '''
    i = atom_index
    new_positions = positions.copy()
    new_displacements = displacements.copy()
    new_lengths = lengths.copy()
    n_atoms = new_positions.shape[0]
    energy_change = 0.0

    new_positions[i] += displacement
    new_displacements[i, i+1: ] = new_positions[i: i+1] - new_positions[i+1: ] 
    new_lengths[i, i+1: ] = np.sum(new_displacements[i, i+1: ] ** 2, axis=1) ** 0.5
    new_displacements[:i, i] = new_positions[:i] - new_positions[i: i+1]
    new_lengths[:i, i] = np.sum(new_displacements[:i, i] ** 2, axis=1) ** 0.5

    if i != 0:
        extension = np.abs(lengths[i-1, i] - equilibrium_bond_length)
        new_extension = np.abs(new_lengths[i-1, i] - equilibrium_bond_length)
        energy_change += 0.5 * spring_constant * (new_extension**2 - extension**2)
    if i != n_atoms-1:
        extension = np.abs(lengths[i, i+1] - equilibrium_bond_length)
        new_extension = np.abs(new_lengths[i, i+1] - equilibrium_bond_length)
        energy_change += 0.5 * spring_constant * (new_extension**2 - extension**2)
    
    #Lennard Jones Potentials ignore 2 nearest neighbours of each atom
    sixpowers = (sigma / lengths[i, i+3: ]) ** 6
    new_sixpowers = (sigma / new_lengths[i, i+3: ]) ** 6
    energy_change += epsilon * (np.sum(new_sixpowers**2 - 2*new_sixpowers) - np.sum(sixpowers**2 - 2*sixpowers))
    if i > 1: #lengths[:i-2, i] returns unintended values if (i-2) is negative
        sixpowers = (sigma / lengths[:i-2, i]) ** 6
        new_sixpowers = (sigma / new_lengths[:i-2, i]) ** 6
        energy_change += epsilon * (np.sum(new_sixpowers**2 - 2*new_sixpowers) - np.sum(sixpowers**2 - 2*sixpowers))

    return energy_change, new_positions, new_displacements, new_lengths

class PotentialEnergyTracker():
    '''
    Utility class for efficiently tracking how total potential energy (bonding + Lennard Jones non-bonding) changes
    when a single atom is displaced.
    '''

    def __init__(self, positions: np.ndarray, epsilon: float, sigma: float, equilibrium_bond_length: float,
                spring_constant: float) -> None:
        'calculates key distances, lengths and total potential energy'
        self.positions = positions
        self.epsilon = epsilon
        self.sigma = sigma
        self.equilibrium_bond_length = equilibrium_bond_length
        self.spring_constant = spring_constant
        self.n_atoms = len(positions)

        # (n_atoms x n_atoms) array where array[i, j] is the displacement going from j to i
        self.displacements = self.positions.reshape(self.n_atoms, 1, 3) - self.positions.reshape(1, self.n_atoms, 3)
        self.lengths = np.sum(self.displacements ** 2, axis=2) ** 0.5

    def get_total_potential_energy(self):
        '''return total potential energy of molecule'''
        _, bonding_potential = get_bonding_interactions(self.positions, self.equilibrium_bond_length, self.spring_constant)
        _, non_bonding_potential = get_non_bonding_interactions(self.positions, self.epsilon, self.sigma)
        return bonding_potential + non_bonding_potential

    def test_displacement(self, atom_index: int, displacement: np.ndarray) -> float:
        '''stores temporary new positions, displacements, and lengths. Returns change in potential energy'''
        result = get_energy_change(self.positions, self.displacements, self.lengths, atom_index, displacement, self.epsilon,
                                   self.sigma, self.equilibrium_bond_length, self.spring_constant)
        energy_change, self.new_positions, self.new_displacements, self.new_lengths = result
        return energy_change

    def accept_last_displacement(self) -> None:
        '''replaces internal positions, displacements, and lengths with values from last test displacement'''
        self.positions = self.new_positions
        self.displacements = self.new_displacements
        self.lengths = self.new_lengths