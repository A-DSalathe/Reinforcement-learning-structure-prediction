#!/usr/bin/env python3
import ase
from ase.cluster.cubic import SimpleCubic
from ase.cluster import Icosahedron, Octahedron

# First example, actual diatomic

surfaces = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
layers = [1, 1, 1]
lc = 1.9
atoms = SimpleCubic("B", surfaces, layers, latticeconstant=lc)
ase.io.write("reference_custom_1.xyz", atoms)

