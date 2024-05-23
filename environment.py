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

    x = np.arange(X)
    y = np.arange(Y)
    z = np.arange(Z)

    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

    coordinates = list(zip(xx.ravel(), yy.ravel(), zz.ravel()))

    return coordinates


def find_distances_to_new_point(array, new_point):
    if (np.array(new_point) >= np.array(array.shape)).any() or (np.array(new_point) < 0).any():
        raise ValueError("New point coordinates are out of the array bounds.")

    existing_points = np.argwhere(array == 1)
    new_point_array = np.array(new_point)
    if existing_points.size == 0:
        return None

    distances = np.sqrt(np.sum((existing_points - new_point_array) ** 2, axis=1))

    min_distance = np.min(distances)
    return min_distance


def path_to_refspectra(ref_spectra_path):
    data = np.loadtxt(ref_spectra_path, skiprows=2)
    return data


def calculate_pairwise_distances(state, resolution):
    atom_positions = np.argwhere(state == 1)
    if len(atom_positions) < 2:
        return np.array([0])  # No meaningful distances to compute

    distances = []
    for i in range(len(atom_positions)):
        for j in range(i + 1, len(atom_positions)):
            dist = np.linalg.norm((atom_positions[i] - atom_positions[j]) * resolution)
            distances.append(dist)
    return np.array(distances)


class Molecule_Environment:
    def __init__(self, n_atoms: int = 2, chemical_symbols: list = ["B"], dimensions=(11, 11, 11),
                 resolution=np.array([0.2, 0.2, 0.2]),
                 ref_spectra_path=op.join(script_dir, op.join('references', 'reference_1_B.dat')), print_spectra=0,
                 min_reward=-10, max_placement_penalty=11, min_placement_penalty=0.1 ,placement_weight=1.0, spectra_weight=10.0):
        self.n_atoms = n_atoms
        self.chemical_symbols = chemical_symbols
        self.dimensions = dimensions
        self.resolution = resolution
        self.state = np.zeros(dimensions)
        self.ref_spectra = path_to_refspectra(ref_spectra_path)
        self.print_spectra = print_spectra
        center = (dimensions[0] // 2, dimensions[1] // 2, dimensions[2] // 2)
        self.state[center] = 1
        self.actions = generate_3d_coordinates(dimensions)
        self.n_actions = len(self.actions)
        self.covalent_radii = 0.9
        self.done = False
        self.chem_symbols = ["B"]
        self.name = "test"
        self.cumulative_reward = 0
        self.spectra = None
        self.min_reward = min_reward
        self.max_placement_penalty = max_placement_penalty
        self.min_placement_penalty = min_placement_penalty
        self.placement_weight = placement_weight
        self.spectra_weight = spectra_weight
        self.rewards = []

    def __str__(self) -> str:
        return f"Molecule Environment(with {self.n_atoms} atoms, in the workspace going from {[0, 0, 0]} to {self.dimensions / self.resolution}, with resolution={self.resolution})"

    def get_actions(self):
        return self.actions

    def reset(self):
        dimensions = self.dimensions
        self.state = np.zeros(dimensions)
        self.done = False
        center = (dimensions[0] // 2, dimensions[1] // 2, dimensions[2] // 2)
        self.state[center] = 1
        self.actions = self.get_actions()
        self.chem_symbols = ["B"]
        self.cumulative_reward = 0
        self.rewards = []
        return self.state

    def step(self, action):
        if not self.done:
            place_atom = (self.state[action] == 0)
            if (self.state.sum() == self.n_atoms - 1) and place_atom:
                self.done = True

            placement_reward = self.get_reward_placement(action)

            if place_atom:
                self.state[action] = 1
                self.chem_symbols.append("B")
            self.cumulative_reward += placement_reward
            self.rewards.append(placement_reward)
        else:
            placement_reward = 0

        if self.done and self.cumulative_reward > self.min_reward:
            spectra_reward = -self.diff_spectra()
            print("spectra reward in if", spectra_reward)
        else:
            spectra_reward = -10

        print(placement_reward)
        print("spectra reward", spectra_reward)
        print("placement reward", placement_reward)
        combined_reward = (self.placement_weight * self.cumulative_reward) + (self.spectra_weight * spectra_reward)
        return self.state, combined_reward, self.done

    def get_reward_placement(self, action):
        min_distance = find_distances_to_new_point(self.state, action) * self.resolution[0]
        reward = 0
        print("min distance", min_distance)

        lower_bound = 0.5 * self.covalent_radii
        upper_bound = 1.5 * self.covalent_radii

        if min_distance == 0:
            reward = -self.max_placement_penalty
        elif min_distance < lower_bound:
            # Linear penalty for being too close, scaled between min and max penalty
            reward = -self.min_placement_penalty - (self.max_placement_penalty - self.min_placement_penalty) * (
                        (lower_bound - min_distance) / lower_bound)
        elif min_distance > upper_bound:
            # Linear penalty for being too far, scaled between min and max penalty
            reward = -self.min_placement_penalty - (self.max_placement_penalty - self.min_placement_penalty) * (
                        (min_distance - upper_bound) / (max(self.dimensions) - upper_bound))
        else:
            reward = 0

        return reward

    def diff_spectra(self):
        ref_spectra_y = self.ref_spectra[:, 1]
        atom_pos = np.where(self.state == 1)
        coords_atom = list(zip(*atom_pos))
        spectra = spectra_from_arrays(positions=np.array(coords_atom) * self.resolution,
                                      chemical_symbols=self.chem_symbols, name=self.name, writing=False,
                                      verbosity=self.print_spectra)
        self.spectra = spectra
        spectra_y = spectra[:, 1]
        return np.linalg.norm(spectra_y - ref_spectra_y, ord=2) * 10 ** 6

    def sample_action(self):
        actions = self.actions
        return actions[np.random.randint(0, len(actions))]

    def render(self):
        print(self.state)

    def index_action(self, action):
        return self.actions.index(action)

    def action_index(self, index):
        return self.actions[index]

    def get_state_features(self):
        distances = calculate_pairwise_distances(self.state, self.resolution)
        return distances


if __name__ == "__main__":
    dimensions = (11, 11, 11)
    resolution = np.array([0.2, 0.2, 0.2])
    env = Molecule_Environment(dimensions=dimensions, resolution=resolution)
    print(env)
    state_flatten = env.state.flatten()
    print(state_flatten)
    possible_actions = env.get_actions()
    env.reset()
    action = env.sample_action()
    print(action)

    state, reward, done = env.step(action)
    print(state)
    print(reward)
    print(done)