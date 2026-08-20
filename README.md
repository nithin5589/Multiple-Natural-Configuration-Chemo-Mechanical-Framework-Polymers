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

The codes in this repository use the legacy DOLFIN interface and should not be directly confused with the newer FEniCSx/DOLFINx framework.

A typical FEniCS installation can be verified in Python using:

```python
from dolfin import *
