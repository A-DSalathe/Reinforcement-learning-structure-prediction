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

class Simple_Environment:
    def __init__(self, n_atoms: int = 2, chemical_symbols: list = ["B"], dimensions = (5,5,5), resolution=np.array([0.4,0.4,0.4]), ref_spectra_path = op.join(script_dir,op.join('references','reference_1_B.dat'))):
        self.n_atoms = n_atoms
        self.chemical_symbols = chemical_symbols
        self.dimensions = dimensions
        self.resolution = resolution
        self.state = np.zeros(dimensions)
        self.ref_spectra = path_to_refspectra(ref_spectra_path)
        # self.n_state = math.comb(dimensions[0]*dimensions[1]*dimensions[2]-1, n_atoms-1)
        center = (dimensions[0]//2, dimensions[1]//2, dimensions[2]//2)
        self.state[center] = 1
        self.actions  = generate_3d_coordinates(dimensions)
        self.n_actions = len(self.actions)
        self.covalent_radii = 0.4
        self.done = False
        self.chem_symbols = ["B"]
        self.name = "test"
        self.cumulative_reward = 0

    def get_actions(self):
        return self.actions

    def reset(self):
        dimensions = self.dimensions
        self.state = np.zeros(dimensions)
        self.done = False
        self.state[dimensions[0]//2, dimensions[1]//2, dimensions[2]//2] = 1
        self.actions  = self.get_actions()
        self.chem_symbols = ["B"]
        return self.state

    def step(self, action, verbose=False):
        if (self.state.sum() == self.n_atoms-1) and (self.state[action] == 0):
            self.done = True
        if verbose:
            return self.get_reward(action, verbose=True)
        reward = self.get_reward(action)
        
        self.cumulative_reward += reward
        self.state[action] = 1
        self.actions = self.get_actions()
        self.chem_symbols.append("B")
        
        return self.state, reward

    def get_reward(self, action, verbose=False):
        min_distance = find_distances_to_new_point(self.state, action)
        reward = self.cumulative_reward
        
        # Define the acceptable distance range
        lower_bound = 0.5 * self.covalent_radii / self.resolution[0]
        upper_bound = 1.5 * self.covalent_radii / self.resolution[0]
        
        # Smooth penalty function
        if min_distance == 0:
            reward = 0
        elif min_distance < lower_bound:
            reward = -np.exp(-100 * (min_distance - lower_bound))  # Smooth penalty for being too close
        elif min_distance > upper_bound:
            reward = -np.exp(100 * (min_distance - upper_bound))  # Smooth penalty for being too far
        else:
            reward = 0  # Reward close to 0 within the acceptable range

        if self.done and not verbose:
            reward += -self.diff_spectra()
        elif self.done and verbose:
            return self.diff_spectra(verbose=True)
        return reward
    
    def diff_spectra(self, verbose=False):
        # Compute the spectra of the current state
        #self.state = spectra_from_arrays()
        ref_spectra_y = self.ref_spectra[:,1]
        atom_pos = np.where(self.state == 1)
        coords_atom = list(zip(*atom_pos))
        # Compute the difference between the current state spectra and the reference spectra
        spectra = spectra_from_arrays(positions=np.array(coords_atom)*self.resolution, chemical_symbols=self.chem_symbols, name=self.name, writing=False)
        spectra_y = spectra[:,1]
        if verbose:
            return np.linalg.norm(spectra_y - ref_spectra_y, ord=2), ref_spectra_y, spectra_y
        else:
            return np.linalg.norm(spectra_y - ref_spectra_y, ord=2)

    # def encoded_action(self, action):
    #     return np.ravel_multi_index(action, self.state.shape)

    def render(self):
        print(self.state)



if __name__ == "__main__":

    env = Simple_Environment()
    # print(env.state)
    possible_actions = env.get_actions()
    # print(possible_actions[27])
    reward, spectra_ref, spectra1 = env.step(possible_actions[0], verbose=True)
    print(reward)
    # print(state)
    # print(reward)
    # state, reward = env.step(possible_actions[0])
    # print(state)
    # print(reward)
    # name = 'bob'
    # resolution = np.array([0.1,0.1,0.1])   
    # num_coordinates = 10
    # coords_atom = np.random.randint(0, 10, size=(num_coordinates, 3))
    # chem_symbols = ["B"] * num_coordinates
    # spectra = spectra_from_arrays(positions=np.array(coords_atom)*resolution, chemical_symbols=chem_symbols, name=name, writing=False)
    # print(env)









    