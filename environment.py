import numpy as np
import math

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

class Simple_Environment:
    def __init__(self, n_atoms: int = 2, dimensions = (11,11,11), resolution=np.array([0.1,0.1,0.1]), ref_spectra: np.ndarray):
        self.n_atoms = n_atoms
        self.dimensions = dimensions
        self.state = np.zeros(dimensions)
        self.ref_spectra = ref_spectra
        # self.n_state = math.comb(dimensions[0]*dimensions[1]*dimensions[2]-1, n_atoms-1)
        center = np.array(dimensions)//2
        self.state[center] = 1
        zero_indices = np.where(self.state == 0)
        self.actions  = list(zip(*zero_indices))
        self.n_actions = len(self.actions)
        self.covalent_radii = 0.4
        self.done = False

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
        return self.state

    def step(self, action):
        if self.state.count_nonzero() == self.n_atoms-1:
            self.done = True
        reward = self.get_reward(action)
        self.state[action] = 1
        self.actions = self.get_actions()
        
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
        # Compute the difference between the current state spectra and the reference spectra
        return np.linalg.norm(self.state - self.ref_spectra, ord=2)
    # def encoded_action(self, action):
    #     return np.ravel_multi_index(action, self.state.shape)

    def render(self):
        print(self.state)


if __name__ == "__main__":
    dim = (2,3,4)
    test = np.zeros(dim)
    print(test)
    print(test.shape)
    test[1,1,1] = 1
    test[1,2,3] = 1
    where_test = np.where(test == 1)
    print(where_test)
    coords = list(zip(*where_test))
    print(coords)
    test2 = test.flatten()
    print(np.where(test2))
    print(np.inf)
    # env = Simple_Environment()
    # print(env)