import numpy as np
import math
from spectra import spectra_from_arrays
from spectra import test
from ase.vibrations import Infrared
from ase import Atoms
import os
import os.path as op
script_dir = op.dirname(op.realpath(__file__))

def generate_3d_coordinates(shape):
    X, Y, Z = shape
    
    # Create ranges for each dimension
    x = np.arange(X)
    y = np.arange(Y)
    z = np.arange(Z)
    
    # Generate coordinate grids
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    
    # Flatten the grids and zip them into tuples
    coordinates = list(zip(xx.ravel(), yy.ravel(), zz.ravel()))
    
    return coordinates

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
    return min_distance

def path_to_refspectra(ref_spectra_path):
        data = np.loadtxt(ref_spectra_path, skiprows=2)
        return data

class Molecule_Environment:
    def __init__(self, n_atoms: int = 2, chemical_symbols: list = ["B"], dimensions = (7,7,7), resolution=np.array([0.3,0.3,0.3]), ref_spectra_path = op.join(script_dir,op.join('references','reference_1_B.dat')), print_spectra=0):
        self.n_atoms = n_atoms
        self.chemical_symbols = chemical_symbols
        self.dimensions = dimensions
        self.resolution = resolution
        self.state = np.zeros(dimensions)
        self.ref_spectra = path_to_refspectra(ref_spectra_path)
        self.print_spectra = print_spectra
        center = (dimensions[0]//2, dimensions[1]//2, dimensions[2]//2)
        self.state[center] = 1
        self.actions  = generate_3d_coordinates(dimensions)
        self.n_actions = len(self.actions)
        self.covalent_radii = 0.9
        self.done = False
        self.chem_symbols = ["B"]
        self.name = "test"
        self.cumulative_reward = 0
        self.spectra = None

    def __str__(self) -> str:
        return f"Molecule Environment(with {self.n_atoms} atoms, in the workspace going from {[0,0,0]} to {self.dimensions/self.resolution}, with resolution={self.resolution})"

    def get_actions(self):
        return self.actions

    def reset(self):
        dimensions = self.dimensions
        self.state = np.zeros(dimensions)
        self.done = False
        center = (dimensions[0] // 2, dimensions[1] // 2, dimensions[2] // 2)
        self.state[center] = 1
        self.actions  = self.get_actions()
        self.chem_symbols = ["B"]
        return self.state

    def step(self, action):
        if not self.done:
            place_atom = (self.state[action] == 0)
            if (self.state.sum() == self.n_atoms-1) and place_atom:
                self.done = True

            reward = self.get_reward_placement(action)
            # reward = 0

            if place_atom:
                self.state[action] = 1
                self.chem_symbols.append("B")

            if self.done:
                reward -= self.diff_spectra()
            self.cumulative_reward += reward
        else:
            reward = self.cumulative_reward
        
        return self.state, reward, self.done

    def get_reward_placement(self, action):
        min_distance = find_distances_to_new_point(self.state, action)
        reward = self.cumulative_reward
        
        # Define the acceptable distance range
        lower_bound = 0.5 * self.covalent_radii / self.resolution[0]
        upper_bound = 1.5 * self.covalent_radii / self.resolution[0]
        
        # Smooth penalty function
        if min_distance == 0:
            reward = 0
        elif min_distance < lower_bound:
            reward = -np.exp(-10 * (min_distance - lower_bound))  # Smooth penalty for being too close
        elif min_distance > upper_bound:
            reward = -np.exp(10 * (min_distance - upper_bound))  # Smooth penalty for being too far
        else:
            reward = 0  # Reward close to 0 within the acceptable range

        return reward
    
    def diff_spectra(self):
        # Compute the spectra of the current state
        #self.state = spectra_from_arrays()
        ref_spectra_y = self.ref_spectra[:,1]
        atom_pos = np.where(self.state == 1)
        coords_atom = list(zip(*atom_pos))
        #print(coords_atom*self.resolution)
        # print(coords_atom*self.resolution)
        # Compute the difference between the current state spectra and the reference spectra
        spectra = spectra_from_arrays(positions=np.array(coords_atom)*self.resolution, chemical_symbols=self.chem_symbols, name=self.name, writing=False, verbosity=self.print_spectra)
        self.spectra = spectra
        spectra_y = spectra[:,1]
        return np.linalg.norm(spectra_y - ref_spectra_y, ord=2)*10**6
    def sample_action(self):
        actions = self.actions
        return actions[np.random.randint(0, len(actions))]
    
    def render(self):
        print(self.state)
    
    def index_action(self, action):
        return self.actions.index(action)
    
    def action_index(self, index):
        return self.actions[index]



if __name__ == "__main__":

    dimensions = (11,11,11)
    resolution = np.array([0.2,0.2,0.2])
    # env = Molecule_Environment(dimensions=dimensions, resolution=resolution)
    env = Molecule_Environment()
    print(env)
    state_flatten = env.state.flatten()
    print(state_flatten)
    # print(env.state)
    possible_actions = env.get_actions()
    env.reset()
    state, reward, done = env.step((3, 2, 1))
    print(state)
    print(reward)
    print(done)
    env.print_spectra = 1
    # state, reward, done = env.step((7,9,5))
    action = env.sample_action()
    print(action)

    state, reward, done = env.step(action)
    print(state)
    print(reward)
    print(done)








    