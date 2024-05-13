import numpy as np
import math
from spectra import spectra_from_arrays
from spectra import test
from ase.vibrations import Infrared
from ase import Atoms
import os
import os.path as op
script_dir = op.dirname(op.realpath(__file__))

def find_distances_to_new_point(array, new_point):
    # Check if the new point is within the array bounds
    if (np.array(new_point) >= np.array(array.shape)).any() or (np.array(new_point) < 0).any():
        raise ValueError("New point coordinates are out of the array bounds.")

    # Find all existing points where the value is 1
    existing_points = np.argwhere(array == 1)
    
    # Convert new_point to a numpy array for distance calculation
    new_point_array = np.array(new_point)
    
    # Compute distances between the new point and all existing points
    if existing_points.size == 0:
        return None, None  # No existing points to compare against

    distances = np.sqrt(np.sum((existing_points - new_point_array) ** 2, axis=1))
    
    # Find minimum and maximum distances
    min_distance = np.min(distances)
    max_distance = np.max(distances)
    return min_distance, max_distance

def path_to_refspectra(ref_spectra_path):
        data = np.loadtxt(ref_spectra_path, skiprows=2)
        return data

class Simple_Environment:
    def __init__(self, n_atoms: int = 2, chemical_symbols: list = ["B"], dimensions = (11,11,11), resolution=np.array([0.1,0.1,0.1]), ref_spectra_path = op.join(script_dir,op.join('references','reference_1_B.dat'))):
        self.n_atoms = n_atoms
        self.chemical_symbols = chemical_symbols
        self.dimensions = dimensions
        self.resolution = resolution
        self.state = np.zeros(dimensions)
        self.ref_spectra = path_to_refspectra(ref_spectra_path)
        # self.n_state = math.comb(dimensions[0]*dimensions[1]*dimensions[2]-1, n_atoms-1)
        center = np.array(dimensions)//2
        self.state[center] = 1
        zero_indices = np.where(self.state == 0)
        self.actions  = list(zip(*zero_indices))
        self.n_actions = len(self.actions)
        self.covalent_radii = 0.4
        self.done = False
        self.chem_symbols = ["B"]
        self.name = "test"

    def get_actions(self):
        zero_indices = np.where(self.state == 0)
        actions = list(zip(*zero_indices))
        return actions

    def reset(self):
        dimensions = self.dimensions
        self.state = np.zeros(dimensions)
        self.done = False
        self.state[dimensions[0]//2, dimensions[1]//2, dimensions[2]//2] = 1
        self.actions  = self.get_actions()
        self.chem_symbols = ["B"]
        return self.state

    def step(self, action):
        if self.state.count_nonzero() == self.n_atoms-1:
            self.done = True
        reward = self.get_reward(action)
        self.state[action] = 1
        self.actions = self.get_actions()
        self.chem_symbols.append("B")
        
        return self.state, reward

    def get_reward(self,action):
        min_distance, max_distance = find_distances_to_new_point(self.state, action)
        if min_distance <= 0.5*self.covalent_radii:
            reward = -np.inf
        elif max_distance >= 1.5*self.covalent_radii:
            reward = -np.inf
        else:
            reward = 0
        if self.done:
            reward += -self.diff_spectra()
        return reward
    
    def diff_spectra(self):
        # Compute the spectra of the current state
        #self.state = spectra_from_arrays()
        ref_spectra_y = self.ref_spectra[:,1]
        # Compute the difference between the current state spectra and the reference spectra
        spectra = spectra_from_arrays(positions=np.array(self.actions)*self.resolution, chemical_symbols=self.chem_symbols, name=self.name)
        spectra_y = spectra[:,1]
        return np.linalg.norm(spectra_y - ref_spectra_y, ord=2)
    # def encoded_action(self, action):
    #     return np.ravel_multi_index(action, self.state.shape)

    def render(self):
        print(self.state)

    

    def make_spectra(self):
        #doit lire le fichier à chaque fois, il faudrait faire une fonction qui stock dans un array plutôt que faire ça
        #work in progress
        pass
        nanoparticle = ase.io.read('nanoparticle.xyz')
        nanoparticle.calc = XTB3(method="GFN2-xTB", max_iterations=1000)
        ir = Infrared(nanoparticle)
        ir.run()
        energy_range, spectrum = ir.get_spectrum(
            start=0, end=1000, width=10, normalize=True
        )
        ir.write_spectra(f'spectra.dat', start=0, end=1000, width=10)

    def make_particle(self):
        pass
        #create a position file similar to the one we have in references
        # with open('output_file.txt', 'w') as f:
        #     # Write number of atoms
        #     num_atoms = np.sum(self.state)  # Count the number of atoms
        #     f.write(f"{int(num_atoms)}\n")
        #
        #     # Write properties line
        #     f.write("Properties=species:S:1:pos:R:3 pbc=\"F F F\"\n")
        #
        #     # Iterate over the positions in self.state
        #     for i in range(dimensions[0]):
        #         for j in range(dimensions[1]):
        #             for k in range(dimensions[2]):
        #                 # Check if the value is 1, indicating the presence of a "B" atom
        #                 if self.state[i, j, k] == 1:
        #                     # Calculate the position of the atom
        #                     x = i * resolution[0] - 0.475
        #                     y = j * resolution[1] - 0.475
        #                     z = k * resolution[2]
        #                     # Write the atom position to the file
        #                     f.write(f"B {x:.8f} {y:.8f} {z:.8f}\n")



if __name__ == "__main__":
    test_array = test()
    print(test_array[:,1])
    # ref_path = op.join('references', 'reference_1_B.dat')
    # ref_path = op.join(script_dir, ref_path)
    # test_ref = path_to_refspectra(ref_path)
    # print(test_ref)
    # dim = (2,3,4)
    # test = np.zeros(dim)
    # print(test)
    # print(test.shape)
    # test[1,1,1] = 1
    # test[1,2,3] = 1
    # where_test = np.where(test == 1)
    # print(where_test)
    # coords = list(zip(*where_test))
    # print(coords)
    # test2 = test.flatten()
    # print(np.where(test2))
    # print(np.inf)
    # env = Simple_Environment()
    # print(env)