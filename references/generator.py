#!/usr/bin/env python3
import ase
from ase.cluster.cubic import FaceCenteredCubic
from ase.cluster import Icosahedron, Octahedron

# First example, actual diatomic

surfaces = [(1, 0, 1), (1, 1, 0), (1, 1, 1)]
layers = [1, 2, 2]
lc = 1.9
atoms = FaceCenteredCubic("B", surfaces, layers, latticeconstant=lc)
ase.io.write("reference_1_B.xyz", atoms)

# Second example

surfaces = [(1, 0, 1), (1, 1, 0), (1, 1, 1)]
layers = [2, 1, 2]
lc = 2.3
atoms = FaceCenteredCubic("B", surfaces, layers, latticeconstant=lc)
ase.io.write("reference_2_B.xyz", atoms)

# Third example

surfaces = [(1, 0, 1), (1, 1, 0), (1, 1, 1)]
layers = [2, 2, 2]
lc = 2.4
atoms = FaceCenteredCubic("B", surfaces, layers, latticeconstant=lc)
ase.io.write("reference_3_B.xyz", atoms)

# Fourth example

surfaces = [(1, 0, 0), (1, 1, 1), (1, -1, 1)]
layers = [2, 2, -1]
lc = 2.2
atoms = FaceCenteredCubic("B", surfaces, layers, latticeconstant=lc)
ase.io.write("reference_4_B.xyz", atoms)

# Fifth example, truncated octahedron

lc = 2.4
atoms = Octahedron("B", 3, cutoff=1, latticeconstant=lc)
ase.io.write("reference_5_B.xyz", atoms)

# Sixth example, truncated octahedron without perfect symmetry

lc = 2.6
atoms = Octahedron("B", 3, cutoff=1, latticeconstant=lc)
atoms.rattle(stdev=0.05, seed=42)
ase.io.write("reference_6_B.xyz", atoms)


# Seventh example, alloy!

lc = 2.6
atoms = Octahedron(["B", "O"], 3, alloy=True, latticeconstant=lc)
ase.io.write("reference_7_BO.xyz", atoms)


# Eight example, alloy!

lc = 2.5
atoms = Octahedron(["O", "B"], 3, cutoff=1, alloy=True, latticeconstant=lc)
ase.io.write("reference_8_BO.xyz", atoms)

# Ninth example, alloy without symmetry!

lc = 2.7
atoms = Octahedron(["O", "B"], 3, cutoff=1, alloy=True, latticeconstant=lc)
atoms.rattle(stdev=0.1, seed=42)
ase.io.write("reference_9_BO.xyz", atoms)
