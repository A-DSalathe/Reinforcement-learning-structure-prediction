import numpy as np
import math
from spectra import spectra_from_arrays
from spectra import test
from ase.vibrations import Infrared
from ase import Atoms
import os
import os.path as op
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    existing_points = np.argwhere(array != 0)
    # Convert new_point to a numpy array for distance calculation
    new_point_array = np.array(new_point)
    # Compute distances between the new point and all existing points
    if existing_points.size == 0:
        return None, None  # No existing points to compare against

    distances = np.sqrt(np.sum((existing_points - new_point_array) ** 2, axis=1))
    
    # Find minimum distances
    min_distance = np.min(distances)
    return min_distance
def path_to_refspectra(ref_spectra_path):
        data = np.loadtxt(ref_spectra_path, skiprows=2)
        return data
def generate_spherical_shell(inner_radius, outer_radius):
    # Define the center of the sphere in the grid
    inner_radius_ceil = int(np.ceil(inner_radius))
    outer_radius_floor = int(np.floor(outer_radius))
    grid_size = outer_radius_floor * 2 + 1
    cx, cy, cz = grid_size // 2, grid_size // 2, grid_size // 2
    
    # Create a 3D grid of indices
    x = np.arange(grid_size) - cx
    y = np.arange(grid_size) - cy
    z = np.arange(grid_size) - cz
    x, y, z = np.meshgrid(x, y, z, indexing='ij')
    
    # Calculate the squared distance from the center
    distance_squared = x**2 + y**2 + z**2
    
    # Create the spherical shell: points between inner_radius and outer_radius are 1, others are 0
    shell = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)
    shell[(distance_squared >= inner_radius_ceil**2) & (distance_squared <= outer_radius_floor**2)] = 1
    
    return shell
def generate_spherical_shell_coordinates(inner_radius, outer_radius):
    # Define the size of the grid
    grid_range = np.arange(-outer_radius, outer_radius + 1)
    
    # Create a 3D grid of points
    x, y, z = np.meshgrid(grid_range, grid_range, grid_range, indexing='ij')
    
    # Calculate the squared distance from the center
    distance_squared = x**2 + y**2 + z**2
    
    # Find points within the spherical shell
    inside_shell = (distance_squared >= inner_radius**2) & (distance_squared <= outer_radius**2)
    
    # Extract the coordinates of these points
    shell_coordinates = np.column_stack((x[inside_shell], y[inside_shell], z[inside_shell]))
    
    return shell_coordinates
def generate_action(n_atoms, covalent_radii_pixels):
    sphere_coord = generate_spherical_shell_coordinates(covalent_radii_pixels/2, covalent_radii_pixels*1.5)
    n = len(sphere_coord)
    actions = np.hstack((sphere_coord, np.ones((n, 1))))
    if n_atoms > 2:
        for i in range(n_atoms-2):
            coordinates_index = np.hstack((sphere_coord, (i+2)*np.ones((n, 1))))
            actions = np.concatenate((actions, coordinates_index), axis=0)
    return actions

def convert_numpy_where_numpy(array):
    x1 = array[0]
    x2 = array[1]
    x3 = array[2]
    output = np.array([x1[0],x2[0],x3[0]])
    return output

class Molecule_Environment:
    def __init__(self, n_atoms: int = 3, chemical_symbols: list = ["B"], dimensions = (51,51,51), resolution=np.array([0.2,0.2,0.2]), ref_spectra_path = op.join(script_dir,op.join('references','reference_1_B.dat')), print_spectra=0, min_reward=-10, cov_radi=0.9):
        self.n_atoms = n_atoms
        self.chemical_symbols = chemical_symbols
        self.dimensions = dimensions
        self.resolution = resolution
        self.grid = np.zeros(dimensions)
        self.ref_spectra = path_to_refspectra(ref_spectra_path)
        self.print_spectra = print_spectra
        center = (dimensions[0]//2, dimensions[1]//2, dimensions[2]//2)
        self.grid[center] = 1
        self.state = self.grid
        self.covalent_radii = cov_radi
        covalent_radii_pixels = self.covalent_radii / self.resolution[0]
        self.actions = generate_action(n_atoms, covalent_radii_pixels)
        self.done = False
        self.chem_symbols = ["B"]
        self.name = "test"
        self.cumulative_reward = 0
        self.spectra = None
        self.min_reward = min_reward
        self.n_step = 1

    def __str__(self) -> str:
        return f"Molecule Environment(with {self.n_atoms} atoms, in the workspace going from {[0,0,0]} to {self.dimensions/self.resolution}, with resolution={self.resolution})"

    def get_actions(self):
        return self.actions

    def reset(self):
        dimensions = self.dimensions
        self.grid = np.zeros(dimensions)
        self.done = False
        center = (dimensions[0] // 2, dimensions[1] // 2, dimensions[2] // 2)
        self.grid[center] = 1
        self.actions  = self.get_actions()
        self.chem_symbols = ["B"]
        self.cumulative_reward = 0
        self.n_step = 1
        self.state = self.grid
        return self.state

    def step(self, action):
        if not self.done:
            pos = self.action_position(action)
            place_atom = (self.grid[pos[0],pos[1],pos[2]] == 0)
            if (self.grid.sum() == (self.n_atoms-1)*(self.n_atoms)/2) and place_atom:
                self.done = True

            reward = self.get_reward_placement((pos[0],pos[1],pos[2]))
            # reward = 0

            if place_atom:
                self.grid[pos[0],pos[1],pos[2]] = self.n_step+1
                self.chem_symbols.append("B")
                self.n_step +=1
            self.cumulative_reward += reward
            reward_spectra = 0
            if self.done and self.cumulative_reward>self.min_reward:
                reward_spectra = -self.diff_spectra()
                print("diff spectra =", -self.diff_spectra())
                reward += reward_spectra
            self.cumulative_reward += reward_spectra
        else:
            reward = 0
        self.state = self.grid
        return self.state, reward, self.done

    def get_reward_placement(self, action):
        min_distance = find_distances_to_new_point(self.grid, action)
        reward = 0
        
        # Define the acceptable distance range
        lower_bound = 0.5 * self.covalent_radii / self.resolution[0]
        upper_bound = 1.5 * self.covalent_radii / self.resolution[0]
        
        # Smooth penalty function
        if min_distance == 0:
            reward = -2
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
        atom_pos = np.where(self.grid != 0)
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
    def action_position(self,action):
        grid = self.grid
        grid = grid.astype(int)
        number_atoms_grid = len(np.where(grid!=0)[0])
        coord = action[0:3]
        atom_sel = action[3]
        atom_sel = (atom_sel % number_atoms_grid)
        if not atom_sel:
            atom_sel = number_atoms_grid
        coord_atom_sel = np.where(grid==atom_sel)
        coord_atom_sel = convert_numpy_where_numpy(coord_atom_sel)
        # print(coord.shape)
        new_coord = coord + coord_atom_sel
        new_coord = np.clip(new_coord,a_min=0, a_max=self.dimensions[0])
        new_coord = np.round(new_coord).astype(int)
        return new_coord

if __name__ == "__main__":
    env = Molecule_Environment(dimensions=(5,5,5),resolution=np.array([0.5,0.5,0.5]))
    # print(env.actions.shape)
    # print(env.actions)
    # print(env.sample_action())
    done = False
    while not done:
        action = env.sample_action()
        state, reward, done = env.step(action)
        print(reward,done)
    print(state)
    # sphere = generate_spherical_shell(2.5, 3.5)
    # print(sphere)
    # # Extract coordinates of the spherical shell's surface
    # coords = np.argwhere(sphere == 1)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c='b', marker='o')
    # plt.show()