# Multiple-Natural-Configuration-Chemo-Mechanical-Framework-Polymers
Computational codes associated with the paper "A Multiple Natural Configuration Framework for Hydrolytic Degradation in Biodegradable Polymers"

## Prerequisites

The computational codes in this repository are implemented using the legacy FEniCS/DOLFIN finite element library. Users should have Python, FEniCS, and Gmsh installed before running the simulations.

### FEniCS

This repository uses the standard legacy FEniCS framework (`dolfin`) for finite element implementation. Users should install a compatible legacy FEniCS distribution and refer to the official FEniCS documentation for installation instructions, syntax, finite element formulations, and solver options.

FEniCS Project:
https://fenicsproject.org/

FEniCS Documentation:
https://fenicsproject.org/docs/

A typical FEniCS installation can be verified in Python using:

```python
from dolfin import *
```

### Gmsh

Gmsh is an open-source finite element mesh generator used to create computational meshes from CAD geometries.

Gmsh can be downloaded from:

https://gmsh.info/

Official Gmsh documentation:

https://gmsh.info/doc/texinfo/gmsh.html

### Mesh Generation and Conversion for FEniCS

CAD geometries can be created or imported into Gmsh and discretized using an appropriate finite element mesh.

A typical preprocessing workflow is:

1. Create or import the CAD geometry into Gmsh.
2. Define the desired mesh size and element type.
3. Generate the finite element mesh.
4. Export the mesh in Gmsh `.msh` format.
5. Convert the `.msh` file into the legacy FEniCS `.xml` format using `dolfin-convert`.
6. Import the converted mesh into the FEniCS Python script.

The Gmsh mesh can be converted using:

```bash
dolfin-convert model.msh model.xml
```
