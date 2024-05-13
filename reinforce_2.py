import sys
print(sys.path)


import numpy as np
import gym
from gym import spaces
from ase import Atoms
#from tblite.ase import TBLite as XTB3
from ase.vibrations import Infrared
import matplotlib.pyplot as plt

class MoleculeEnv(gym.Env):
    """A simple environment for placing atoms to form molecules."""
    def __init__(self):
        super(MoleculeEnv, self).__init__()
        # Define action space and state space
        self.action_space = spaces.Box(low=np.array([-5.0, -5.0, -5.0]), high=np.array([5.0, 5.0, 5.0]), dtype=np.float32)
        # State space contains positions of atoms; initially just one atom at origin
        self.state = np.array([[0.0, 0.0, 0.0]])

    def step(self, action):
        # Add new atom based on action, which specifies its position
        new_atom_position = np.array([action])
        self.state = np.vstack([self.state, new_atom_position])
        
        # Reward is calculated based on the IR spectrum similarity (placeholder)
        reward = self.calculate_reward()
        done = len(self.state) == 2  # Episode done after placing two atoms
        
        return self.state, reward, done, {}

    def reset(self):
        # Reset the environment state back to the initial state
        self.state = np.array([[0.0, 0.0, 0.0]])
        return self.state

    def calculate_reward(self):
        # Placeholder for IR spectrum comparison logic
        molecule = Atoms('B2', positions=self.state)
        molecule.calc = XTB3(method="GFN2-xTB", max_iterations=1000)
        ir = Infrared(molecule)
        ir.run()
        energy_range, spectrum = ir.get_spectrum(start=0, end=1000, width=10, normalize=True)
        # Assuming that we have a target spectrum to compare with
        # reward = -np.sum((spectrum - target_spectrum)**2)
        reward = np.random.random()  # Temporary random reward
        return reward

    def render(self, mode='human'):
        print("Current molecule configuration:")
        for idx, pos in enumerate(self.state):
            print(f"Boron {idx + 1}: {pos}")
            
    def close(self):
        pass

# Testing the environment
env = MoleculeEnv()
state = env.reset()
for _ in range(1):  # Change this to run more steps
    action = env.action_space.sample()  # Random action
    state, reward, done, info = env.step(action)
    env.render()
    if done:
        break
env.close()
