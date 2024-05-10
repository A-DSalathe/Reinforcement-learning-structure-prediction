import numpy as np


class Environment:
    def __init__(self, n_atoms: int = 10, n_dimensions: int = 3):
        self.n_atoms = n_atoms
        self.n_dimensions = n_dimensions
        self.positions = np.random.rand(n_atoms, n_dimensions)
        self.chemical_symbols = ["B"] * n_atoms

    def reset(self):
        self.positions = np.random.rand(self.n_atoms, self.n_dimensions)
        self.chemical_symbols = ["B"] * self.n_atoms

    def step(self, action: int):
        # This is a dummy function, we will implement the real one later
        return self.positions, 0, False, {}

    def render(self):
        print(self.positions)