# ================================================================
# 3D MULTIPHYSICS MODEL WITH 1/8 SYMMETRY
# Hydrolytic Degradation + Mechanical Response
#
# Fields:
#   Cf  = Fluid concentration
#   N   = Molecular weight
#   Cm  = Monomer concentration
#   deg = Degradation variable
#   ue  = Displacement
# ================================================================

from fenics import *
import numpy as np
import pandas as pd
import os
import time

# ================================================================
# START TIMER
# ================================================================
start_time = time.time()

# ================================================================
# OUTPUT DIRECTORY
# ================================================================
directory = "Cube"
os.makedirs(directory, exist_ok=True)

df = pd.DataFrame(columns=[
    "Time",
    "Conc of Fluid",
    "Mol wt",
    "Conc of Monomers",
    "Degradation",
    "Boundary Conc of Fluid"
])

results = XDMFFile(os.path.join(directory, "Chemo_LE_Results.xdmf"))
results.parameters["flush_output"] = True
results.parameters["functions_share_mesh"] = True

# ================================================================
# GEOMETRY AND MESH
# ================================================================
#
# The example geometry is generated directly in FEniCS.
#
# For a user-defined CAD geometry, replace this section with:
#
# mesh = Mesh("model.xml")
# boundaries = MeshFunction("size_t", mesh, "model_facet_region.xml")
#
# The mesh can be generated using Gmsh and converted to the
# legacy FEniCS XML format using dolfin-convert.
# ================================================================

print("Creating 1/8 mesh with symmetry...")

# USER-DEFINED GEOMETRY CONSTANTS
length_eighth = 10.0
width_eighth = 10.0
height_eighth = 10.0

# USER-DEFINED MESH RESOLUTION
nx, ny, nz = 10, 10, 10

mesh = BoxMesh(
    Point(0.0, 0.0, 0.0),
    Point(length_eighth, width_eighth, height_eighth),
    nx, ny, nz
)

print(f"Mesh: {mesh.num_cells()} cells, {mesh.num_vertices()} vertices")
print(f"Eighth section: x=[0,{length_eighth}], y=[0,{width_eighth}], z=[0,{height_eighth}]")

# ================================================================
# BOUNDARY DEFINITIONS
# ================================================================

class SymmetryPlaneX(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0, DOLFIN_EPS)

class SymmetryPlaneY(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.0, DOLFIN_EPS)

class SymmetryPlaneZ(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[2], 0.0, DOLFIN_EPS)

class ExternalFaceX(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], length_eighth, DOLFIN_EPS)

class ExternalFaceY(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], width_eighth, DOLFIN_EPS)

class ExternalFaceZ(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[2], height_eighth, DOLFIN_EPS)

# ================================================================
# BOUNDARY MARKERS
# ================================================================
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
boundaries.set_all(0)

symmetry_x = SymmetryPlaneX()
symmetry_y = SymmetryPlaneY()
symmetry_z = SymmetryPlaneZ()
external_x = ExternalFaceX()
external_y = ExternalFaceY()
external_z = ExternalFaceZ()

# Boundary IDs:
# 1 = x=0 symmetry plane
# 2 = y=0 symmetry plane
# 3 = z=0 symmetry plane
# 4 = x=L external surface
# 5 = y=W external surface
# 6 = z=H external surface

symmetry_x.mark(boundaries, 1)
symmetry_y.mark(boundaries, 2)
symmetry_z.mark(boundaries, 3)
external_x.mark(boundaries, 4)
external_y.mark(boundaries, 5)
external_z.mark(boundaries, 6)

# Optional boundary verification
n_sym_x = sum(1 for f in facets(mesh) if boundaries[f] == 1)
n_sym_y = sum(1 for f in facets(mesh) if boundaries[f] == 2)
n_sym_z = sum(1 for f in facets(mesh) if boundaries[f] == 3)
n_ext_x = sum(1 for f in facets(mesh) if boundaries[f] == 4)
n_ext_y = sum(1 for f in facets(mesh) if boundaries[f] == 5)
n_ext_z = sum(1 for f in facets(mesh) if boundaries[f] == 6)

print("\nBoundary marking:")
print(f"x=0 symmetry plane (ID 1): {n_sym_x} facets")
print(f"y=0 symmetry plane (ID 2): {n_sym_y} facets")
print(f"z=0 symmetry plane (ID 3): {n_sym_z} facets")
print(f"x={length_eighth} external surface (ID 4): {n_ext_x} facets")
print(f"y={width_eighth} external surface (ID 5): {n_ext_y} facets")
print(f"z={height_eighth} external surface (ID 6): {n_ext_z} facets")

ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
dx = Measure("dx", domain=mesh)

File(os.path.join(directory, "boundaries_eighth.pvd")) << boundaries

# ================================================================
# FUNCTION SPACE
# ================================================================
#
# Mixed field ordering:
# 0 -> Cf
# 1 -> N
# 2 -> Cm
# 3 -> deg
# 4 -> ue
# ================================================================

P1 = FiniteElement("P", tetrahedron, 1)
P2 = VectorElement("P", tetrahedron, 1)

element = MixedElement([
    P1,   # Fluid concentration
    P1,   # Molecular weight
    P1,   # Monomer concentration
    P1,   # Degradation
    P2    # Displacement
])

V = FunctionSpace(mesh, element)
print(f"Function space: {V.dim()} DOFs")

# ================================================================
# SOLUTION AND TEST FUNCTIONS
# ================================================================
u = Function(V)
Cf, N, Cm, deg, ue = split(u)

v = TestFunction(V)
v_Cf, v_N, v_Cm, v_deg, v_u = split(v)

u_n = Function(V)

# ================================================================
# INITIAL CONDITIONS
# ================================================================
#
# USER-DEFINED INITIAL CONDITIONS
# ================================================================
Cf_0 = Expression("0.0", degree=1)
N_0_initial = Expression("", degree=1)    # Enter your initial molecular weight
Cm_0 = Expression("0.0", degree=1)
deg_0 = Expression("0.0", degree=1)
ue_0 = Expression(("0.0", "0.0", "0.0"), degree=1)

u_0 = Expression(
    ("Cf_n", "N_n", "Cm_n", "deg_n", "ue_n[0]", "ue_n[1]", "ue_n[2]"),
    degree=1,
    Cf_n=Cf_0,
    N_n=N_0_initial,
    Cm_n=Cm_0,
    deg_n=deg_0,
    ue_n=ue_0
)

u_n.interpolate(u_0)
Cf_n, N_n, Cm_n, deg_n, ue_n = split(u_n)

# ================================================================
# USER-DEFINED MODEL CONSTANTS
# ================================================================
#
# Declare or replace all chemical and transport constants here.
# Keep units consistent throughout the model.
# ================================================================

# Fluid diffusion parameters
Df = Constant()
Kf = Constant()

# Chain-scission / degradation parameters
k1 = Constant()
k2 = Constant()
k3 = Constant()
n_pow = Constant()

# Monomer diffusion coefficient
Dm = Constant()

# Initial molecular weight
N0 = Constant()

# ================================================================
# USER-DEFINED MECHANICAL CONSTANTS
# ================================================================
#
# YM0 = Young's modulus of the original network
# nu0 = Poisson's ratio of the original network
# YMf = Young's modulus of the degraded/newly formed network
# nuf = Poisson's ratio of the degraded/newly formed network
# ================================================================

YM0 = Constant()
nu0 = Constant()

lam0 = YM0 * nu0 / ((1.0 + nu0) * (1.0 - 2.0 * nu0))
mu0 = YM0 / (2.0 * (1.0 + nu0))

YMf = Constant()
nuf = Constant()

lamf = YMf * nuf / ((1.0 + nuf) * (1.0 - 2.0 * nuf))
muf = YMf / (2.0 * (1.0 + nuf))

# ================================================================
# USER-DEFINED SWELLING CONSTANT
# ================================================================
beta_swelling = Constant()

# ================================================================
# STRAIN DEFINITIONS
# ================================================================
def eps(u):
    return 0.5 * (grad(u) + grad(u).T)

def eps_sw(Cm):
    return beta_swelling * Cm * Identity(3)

# ================================================================
# HISTORY STORAGE
# ================================================================
eps_sw_history = []
zeta_history = []

# ================================================================
# RESIDUAL STRESS CONTRIBUTION
# ================================================================
def sigma_residual_only(ue, Cm, zeta):
    """
    Residual stress contribution generated by the mismatch between
    total deformation and swelling deformation as the material evolves.
    """
    eps_total = eps(ue)
    eps_sw_current = eps_sw(Cm)
    eps_residual = eps_total - eps_sw_current

    lam_eff = (1.0 - zeta) * lam0 + zeta * lamf
    mu_eff = (1.0 - zeta) * mu0 + zeta * muf

    sigma_residual = (
        lam_eff * tr(eps_residual) * Identity(3)
        + 2.0 * mu_eff * eps_residual
    )
    return sigma_residual

# ================================================================
# HISTORY STRESS CONTRIBUTION
# ================================================================
def sigma_history_only(eps_sw_history, zeta_history):
    """
    Stress contribution associated with the evolving natural
    configurations generated during degradation.
    """
    sigma_history = Constant(0.0) * Identity(3)

    for i in range(1, len(zeta_history)):
        delta_zeta = zeta_history[i] - zeta_history[i - 1]
        eps_sw_i = eps_sw_history[i]

        sigma_history += delta_zeta * (
            lamf * tr(eps_sw_i) * Identity(3)
            + 2.0 * muf * eps_sw_i
        )

    return sigma_history

# ================================================================
# TOTAL STRESS
# ================================================================
def sigma(ue, Cm, eps_sw_history, zeta_history):
    """
    Total stress = residual stress + history stress.
    """
    sigma_residual = sigma_residual_only(ue, Cm, zeta_history[-1])
    sigma_history = sigma_history_only(eps_sw_history, zeta_history)
    return sigma_residual + sigma_history

# ================================================================
# INITIAL HISTORY STATE
# ================================================================
zeta_initial = Constant(0.0)
zeta_history.append(zeta_initial)

eps_sw_initial = eps_sw(Cm_n)
eps_sw_history.append(eps_sw_initial)

# ================================================================
# BOUNDARY CONDITIONS
# ================================================================

# Symmetry conditions
bc_sym_x = DirichletBC(V.sub(4).sub(0), Constant(0.0), boundaries, 1)
bc_sym_y = DirichletBC(V.sub(4).sub(1), Constant(0.0), boundaries, 2)
bc_sym_z = DirichletBC(V.sub(4).sub(2), Constant(0.0), boundaries, 3)

# USER-DEFINED FLUID BOUNDARY VALUE
Cf_boundary_value = Constant(1.0)

bc_fluid_x = DirichletBC(V.sub(0), Cf_boundary_value, boundaries, 4)
bc_fluid_y = DirichletBC(V.sub(0), Cf_boundary_value, boundaries, 5)
bc_fluid_z = DirichletBC(V.sub(0), Cf_boundary_value, boundaries, 6)

# USER-DEFINED BODY FORCE
f_body = Constant((0.0, 0.0, 0.0))

# ================================================================
# USER-DEFINED TIME PARAMETERS
# ================================================================
T_final = 24.0 * 20.0   # 20 days
dt = 6.0                # 6 hours
t = 0.0

# ================================================================
# GOVERNING EQUATIONS
# ================================================================

# Fluid diffusion
F_fluid = (
    ((Cf - Cf_n) / dt) * v_Cf * dx
    + Df * dot(grad(Cf), grad(v_Cf)) * dx
    + Kf * Cf * v_Cf * dx
)

# Molecular-weight reduction / chain scission
F_molecular_weight = (
    ((N - N_n) / dt) * v_N * dx
    + k1 * Cf * N * v_N * dx
    + (k2 * N * Cm**n_pow / (1.0 + k3 * Cm**n_pow)) * v_N * dx
)

# Monomer transport
F_monomer = (
    ((Cm - Cm_n) / dt) * v_Cm * dx
    + Dm * dot(grad(Cm), grad(v_Cm)) * dx
    - (1.0 / N0) * k1 * Cf * N * v_Cm * dx
    - (1.0 / N0) * (k2 * N * Cm**n_pow / (1.0 + k3 * Cm**n_pow)) * v_Cm * dx
)

# Degradation variable
F_degradation = (
    deg * v_deg * dx
    - (1.0 - N / N0) * v_deg * dx
)

Fnonmech = F_fluid + F_molecular_weight + F_monomer + F_degradation

# Mechanical problem
sigma_total = sigma(ue, Cm, eps_sw_history, zeta_history)

Fmech = (
    inner(sigma_total, eps(v_u)) * dx
    - dot(f_body, v_u) * dx
)

# Complete coupled residual
F = Fnonmech + Fmech

# Jacobian
J = derivative(F, u)

# ================================================================
# BOUNDARY CONDITION LIST
# ================================================================
ubcs = [
    bc_fluid_x,
    bc_fluid_y,
    bc_fluid_z,
    bc_sym_x,
    bc_sym_y,
    bc_sym_z
]

print("\nBoundary conditions:")
print("ux = 0 at x = 0")
print("uy = 0 at y = 0")
print("uz = 0 at z = 0")
print("Cf = 1.0 on the three exposed external surfaces")

# ================================================================
# NONLINEAR SOLVER
# ================================================================
#
# USER-DEFINED SOLVER SETTINGS
# ================================================================
problem = NonlinearVariationalProblem(F, u, ubcs, J)
solver = NonlinearVariationalSolver(problem)

prm = solver.parameters
prm["nonlinear_solver"] = "newton"
prm["newton_solver"]["linear_solver"] = "mumps"
prm["newton_solver"]["maximum_iterations"] = 50
prm["newton_solver"]["relative_tolerance"] = 1e-5
prm["newton_solver"]["absolute_tolerance"] = 1e-6
prm["newton_solver"]["relaxation_parameter"] = 0.8
prm["newton_solver"]["error_on_nonconvergence"] = False
prm["newton_solver"]["report"] = True

# ================================================================
# TIME LOOP
# ================================================================
output_time = 0.0
converged = True

print("\n" + "=" * 70)
print("STARTING TIME LOOP - 1/8 SYMMETRY MODEL")
print("=" * 70 + "\n")

while t <= T_final and converged:

    try:
        solver.solve()
        converged = True

    except RuntimeError as error:
        print(f"Solver failed at t = {t:.2f} hours")
        print(error)
        converged = False
        break

    # Obtain converged field solutions
    Cf_sol, N_sol, Cm_sol, deg_sol, ue_sol = u.split()

    # Update degradation and swelling history
    domain_volume = assemble(Constant(1.0) * dx)
    zeta_n = assemble(deg_sol * dx) / domain_volume
    zeta_history.append(zeta_n)

    eps_sw_n = eps_sw(Cm_sol)
    eps_sw_history.append(eps_sw_n)

    # ============================================================
    # OUTPUT
    # ============================================================
    if np.isclose(t, output_time, atol=1e-8):

        # Chemical fields
        Cf_sol.rename("Conc. of Fluid, Cf", "")
        N_sol.rename("Mol. wt., N", "")
        Cm_sol.rename("Conc. of Monomers, Cm", "")
        deg_sol.rename("Degradation, deg", "")

        results.write(Cf_sol, t)
        results.write(N_sol, t)
        results.write(Cm_sol, t)
        results.write(deg_sol, t)

        # Domain averages
        Cf_avg = assemble(Cf_sol * dx) / domain_volume
        N_avg = assemble(N_sol * dx) / domain_volume
        Cm_avg = assemble(Cm_sol * dx) / domain_volume
        deg_avg = assemble(deg_sol * dx) / domain_volume

        # Average fluid concentration on exposed surfaces
        external_area = (
            assemble(Constant(1.0) * ds(4))
            + assemble(Constant(1.0) * ds(5))
            + assemble(Constant(1.0) * ds(6))
        )

        Cf_boundary = (
            assemble(Cf_sol * ds(4))
            + assemble(Cf_sol * ds(5))
            + assemble(Cf_sol * ds(6))
        ) / external_area

        # Displacement
        ue_sol.rename("Displacement", "")
        results.write(ue_sol, t)

        # Tensor spaces
        Vsig = TensorFunctionSpace(mesh, "P", 1)
        Ve = TensorFunctionSpace(mesh, "P", 1)

        # Strain
        eps_proj = project(eps(ue_sol), Ve)
        eps_proj.rename("Strain", "")
        results.write(eps_proj, t)

        # Total stress
        sigma_total_output = sigma(ue_sol, Cm_sol, eps_sw_history, zeta_history)
        sig_total_proj = project(sigma_total_output, Vsig)
        sig_total_proj.rename("Stress_Total", "")
        results.write(sig_total_proj, t)

        # Residual stress
        sigma_residual = sigma_residual_only(ue_sol, Cm_sol, zeta_history[-1])
        sig_residual_proj = project(sigma_residual, Vsig)
        sig_residual_proj.rename("Stress_Residual", "")
        results.write(sig_residual_proj, t)

        # History stress
        sigma_history = sigma_history_only(eps_sw_history, zeta_history)
        sig_history_proj = project(sigma_history, Vsig)
        sig_history_proj.rename("Stress_History", "")
        results.write(sig_history_proj, t)

        # Save averaged values
        df.loc[len(df)] = [
            t / 24.0,
            float(Cf_avg),
            float(N_avg),
            float(Cm_avg),
            float(deg_avg),
            float(Cf_boundary)
        ]

        output_time += 6.0

    # Update previous time step
    u_n.assign(u)

    print(
        f"Time = {t:.2f} hr, "
        f"Total Time = {T_final:.2f} hr, "
        f"Average Degradation = {float(zeta_n):.6f}"
    )

    t += dt

# ================================================================
# SAVE RESULTS
# ================================================================
if converged:
    df.to_excel(
        os.path.join(directory, "FEA_LE_Cube_lowDm.xlsx"),
        index=False
    )

    print("\n" + "=" * 70)
    print("Simulation completed successfully.")
    print("=" * 70)

else:
    print("\n" + "=" * 70)
    print("Simulation failed to converge.")
    print("=" * 70)

results.close()

# ================================================================
# EXECUTION TIME
# ================================================================
end_time = time.time()
elapsed_time = end_time - start_time

print("\n" + "=" * 60)
print(f"Total execution time: {elapsed_time:.2f} seconds")
print(f"Total execution time: {elapsed_time / 60.0:.2f} minutes")
print(f"Total execution time: {elapsed_time / 3600.0:.2f} hours")
print("=" * 60)

# ================================================================
# MODEL SUMMARY
# ================================================================
print("\n" + "=" * 70)
print("1/8 SYMMETRY MODEL SUMMARY")
print("=" * 70)
print("Fields:")
print("  Fluid concentration, Cf")
print("  Molecular weight, N")
print("  Monomer concentration, Cm")
print("  Degradation, deg")
print("  Displacement, ue")

print("\nMechanical outputs:")
print("  Total stress")
print("  Residual stress")
print("  History stress")

print("\nSymmetry planes:")
print("  x = 0: ux = 0")
print("  y = 0: uy = 0")
print("  z = 0: uz = 0")

print("\nExternal surfaces: Cf = 1.0")
print("=" * 70)
