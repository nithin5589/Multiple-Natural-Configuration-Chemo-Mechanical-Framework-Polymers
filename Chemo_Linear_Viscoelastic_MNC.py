# ================================================================
# 3D HALF-SPHERE MULTIPHYSICS MODEL
# Hydrolytic Degradation + Linear Viscoelastic Response
#
# Fields:
#   Cf   = Fluid concentration
#   N    = Molecular weight
#   Cm   = Monomer concentration
#   zeta = Degradation variable, 1 - N/N0
#   ue   = Displacement
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
directory = "Sphere"
os.makedirs(directory, exist_ok=True)

df = pd.DataFrame(columns=[
    "Time",
    "Conc of Fluid",
    "Mol wt",
    "Conc of Monomers",
    "Zeta",
    "Boundary Conc of Fluid"
])

results = XDMFFile(os.path.join(directory, "LVE_HalfSphere_Spherical.xdmf"))
results.parameters["flush_output"] = True
results.parameters["functions_share_mesh"] = True

# ================================================================
# USER-DEFINED PARAMETERS
# ================================================================
#
# Enter the required values before running the simulation.
# Keep all units consistent throughout the model.
# ================================================================

# Initial conditions
Cf_initial_value = None
Cm_initial_value = None
N0_value = None

# Fluid transport
Df_value = None
Kf_value = None

# Hydrolysis / chain scission
k1_value = None
k2_value = None
k3_value = None
n_pow_value = None

# Monomer transport
Dm_value = None

# Virgin material viscoelastic properties
E0_inf_value = None
E0_1_value = None
tau0_value = None
nu0_value = None

# Degraded material viscoelastic properties
Ef_inf_value = None
Ef_1_value = None
tauf_value = None
nuf_value = None

# Swelling coefficient
beta_value = None

# Fluid concentration on exposed surface
Cf_boundary_value = None

# Time parameters
T_final_value = None
dt_value = None

# ================================================================
# CHECK USER INPUT
# ================================================================
parameters = {
    "Cf_initial": Cf_initial_value,
    "Cm_initial": Cm_initial_value,
    "N0": N0_value,
    "Df": Df_value,
    "Kf": Kf_value,
    "k1": k1_value,
    "k2": k2_value,
    "k3": k3_value,
    "n_pow": n_pow_value,
    "Dm": Dm_value,
    "E0_inf": E0_inf_value,
    "E0_1": E0_1_value,
    "tau0": tau0_value,
    "nu0": nu0_value,
    "Ef_inf": Ef_inf_value,
    "Ef_1": Ef_1_value,
    "tauf": tauf_value,
    "nuf": nuf_value,
    "beta": beta_value,
    "Cf_boundary": Cf_boundary_value,
    "T_final": T_final_value,
    "dt": dt_value
}

missing = [name for name, value in parameters.items() if value is None]

if missing:
    raise ValueError(
        "Enter values for the following parameters before running the model: "
        + ", ".join(missing)
    )

# ================================================================
# CONSTANTS
# ================================================================
Cf_initial = Constant(Cf_initial_value)
Cm_initial = Constant(Cm_initial_value)
N0 = Constant(N0_value)

Df = Constant(Df_value)
Kf = Constant(Kf_value)

k1 = Constant(k1_value)
k2 = Constant(k2_value)
k3 = Constant(k3_value)
n_pow = Constant(n_pow_value)

Dm = Constant(Dm_value)

beta = Constant(beta_value)
Cf_external = Constant(Cf_boundary_value)

T_final = float(T_final_value)
dt = float(dt_value)

# ================================================================
# MESH
# ================================================================
#
# The mesh files are stored in the MeshFiles directory.
#
# sphereHalf.xml
# sphereHalf_facet_region.xml
#
# The mesh can be generated in Gmsh and converted to the legacy
# FEniCS XML format using dolfin-convert.
# ================================================================

print("Loading half-sphere mesh...")

mesh = Mesh("MeshFiles/sphereHalf.xml")
boundaries = MeshFunction(
    "size_t",
    mesh,
    "MeshFiles/sphereHalf_facet_region.xml"
)

dx = Measure("dx", domain=mesh)
ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

n_sym = sum(1 for f in facets(mesh) if boundaries[f] == 1)
n_cf = sum(1 for f in facets(mesh) if boundaries[f] == 2)

print(f"Mesh: {mesh.num_cells()} cells, {mesh.num_vertices()} vertices")
print(f"Symmetry plane (ID 1): {n_sym} facets")
print(f"Fluid surface (ID 2): {n_cf} facets")

# ================================================================
# FUNCTION SPACE
# ================================================================
#
# Mixed field ordering:
#   0 -> Cf
#   1 -> N
#   2 -> Cm
#   3 -> ue
# ================================================================

P1 = FiniteElement("P", tetrahedron, 1)
P2 = VectorElement("P", tetrahedron, 1)

element = MixedElement([
    P1,   # Fluid concentration
    P1,   # Molecular weight
    P1,   # Monomer concentration
    P2    # Displacement
])

V = FunctionSpace(mesh, element)

u = Function(V)
Cf, N, Cm, ue = split(u)

v = TestFunction(V)
v_Cf, v_N, v_Cm, v_u = split(v)

u_n = Function(V)

print(f"Function space: {V.dim()} DOFs")

# ================================================================
# INITIAL CONDITIONS
# ================================================================

Cf_0 = Expression("value", value=float(Cf_initial_value), degree=1)
N_0 = Expression("value", value=float(N0_value), degree=1)
Cm_0 = Expression("value", value=float(Cm_initial_value), degree=1)
ue_0 = Expression(("0.0", "0.0", "0.0"), degree=1)

u_0 = Expression(
    ("Cf_n", "N_n", "Cm_n", "ue_n[0]", "ue_n[1]", "ue_n[2]"),
    degree=1,
    Cf_n=Cf_0,
    N_n=N_0,
    Cm_n=Cm_0,
    ue_n=ue_0
)

u_n.interpolate(u_0)
Cf_n, N_n, Cm_n, ue_n = split(u_n)

# ================================================================
# VISCOELASTIC MATERIAL PROPERTIES
# ================================================================
#
# Virgin material:
#
# E(t) = E0_inf + E0_1 exp(-t/tau0)
#
# Degraded material:
#
# E(t) = Ef_inf + Ef_1 exp(-t/tauf)
# ================================================================

E0_inf = E0_inf_value
E0_1 = E0_1_value
tau0 = tau0_value
nu0 = nu0_value

Ef_inf = Ef_inf_value
Ef_1 = Ef_1_value
tauf = tauf_value
nuf = nuf_value

def E_to_mu(E, nu):
    return E / (2.0 * (1.0 + nu))

def E_to_lambda(E, nu):
    return E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

# Instantaneous Lamé parameters
mu0_0 = E_to_mu(E0_inf + E0_1, nu0)
lam0_0 = E_to_lambda(E0_inf + E0_1, nu0)

muf_0 = E_to_mu(Ef_inf + Ef_1, nuf)
lamf_0 = E_to_lambda(Ef_inf + Ef_1, nuf)

# ================================================================
# RELAXATION FUNCTIONS
# ================================================================

def mus_0(t, s):
    return E_to_mu(E0_1 / tau0 * exp(-(t - s) / tau0), nu0)

def lamdas_0(t, s):
    return E_to_lambda(E0_1 / tau0 * exp(-(t - s) / tau0), nu0)

def mus_f(t, s):
    return E_to_mu(Ef_1 / tauf * exp(-(t - s) / tauf), nuf)

def lamdas_f(t, s):
    return E_to_lambda(Ef_1 / tauf * exp(-(t - s) / tauf), nuf)

# ================================================================
# STRAIN FUNCTIONS
# ================================================================

def epsilon(ue):
    return 0.5 * (grad(ue) + grad(ue).T)

def epsilon_sw(Cm):
    return beta * Cm * Identity(3)

def epsilon_residual(ue, Cm):
    return epsilon(ue) - epsilon_sw(Cm)

# ================================================================
# RESIDUAL STRESS
# ================================================================

def sigma_residual_only(ue_current, Cm_current, zeta_current):
    """
    Instantaneous residual stress contribution.
    """

    eps_residual = epsilon_residual(ue_current, Cm_current)

    sigma_residual = (1.0 - zeta_current) * (
        2.0 * mu0_0 * eps_residual
        + lam0_0 * tr(eps_residual) * Identity(3)
    ) + zeta_current * (
        2.0 * muf_0 * eps_residual
        + lamf_0 * tr(eps_residual) * Identity(3)
    )

    return sigma_residual

# ================================================================
# HISTORY STRESS
# ================================================================

def sigma_history_only(
    u_history,
    zeta_history,
    delta_zeta_history,
    t_current,
    t0,
    dt
):

    sigma_history = Constant(0.0) * Identity(3)

    if len(u_history) == 0:
        return sigma_history

    # First stored configuration
    u_first = u_history[0]
    _, _, Cm_first, ue_first = u_first.split(True)

    eps_residual_first = epsilon_residual(ue_first, Cm_first)
    zeta_first = zeta_history[0]

    sigma_history -= Constant(dt / 2.0) * (
        (1.0 - zeta_first) * (
            2.0 * mus_0(t_current, t0) * eps_residual_first
            + lamdas_0(t_current, t0)
            * tr(eps_residual_first) * Identity(3)
        )
        + zeta_first * (
            2.0 * mus_f(t_current, t0) * eps_residual_first
            + lamdas_f(t_current, t0)
            * tr(eps_residual_first) * Identity(3)
        )
    )

    # Previous configurations
    for j in range(1, len(u_history)):
        tj = t0 + j * dt

        u_j = u_history[j]
        _, _, Cm_j, ue_j = u_j.split(True)

        eps_residual_j = epsilon_residual(ue_j, Cm_j)
        zeta_j = zeta_history[j]

        sigma_history -= Constant(dt) * (
            (1.0 - zeta_j) * (
                2.0 * mus_0(t_current, tj) * eps_residual_j
                + lamdas_0(t_current, tj)
                * tr(eps_residual_j) * Identity(3)
            )
            + zeta_j * (
                2.0 * mus_f(t_current, tj) * eps_residual_j
                + lamdas_f(t_current, tj)
                * tr(eps_residual_j) * Identity(3)
            )
        )

    # ------------------------------------------------------------
    # SWELLING HISTORY
    # ------------------------------------------------------------

    for j in range(1, len(delta_zeta_history)):

        delta_zeta_j = delta_zeta_history[j]

        if abs(delta_zeta_j) < 1e-10:
            continue

        tj = t0 + j * dt

        u_j = u_history[j]
        _, _, Cm_j, _ = u_j.split(True)

        eps_sw_j = epsilon_sw(Cm_j)

        sigma_sw_inst = (
            2.0 * muf_0 * eps_sw_j
            + lamf_0 * tr(eps_sw_j) * Identity(3)
        )

        sigma_sw_relax = Constant(0.0) * Identity(3)

        sigma_sw_relax += Constant(dt / 2.0) * (
            2.0 * mus_f(t_current, tj) * eps_sw_j
            + lamdas_f(t_current, tj)
            * tr(eps_sw_j) * Identity(3)
        )

        for k in range(j + 1, len(u_history)):
            tk = t0 + k * dt

            sigma_sw_relax += Constant(dt) * (
                2.0 * mus_f(t_current, tk) * eps_sw_j
                + lamdas_f(t_current, tk)
                * tr(eps_sw_j) * Identity(3)
            )

        if len(u_history) > j:
            sigma_sw_relax += Constant(dt / 2.0) * (
                2.0 * mus_f(t_current, t_current) * eps_sw_j
                + lamdas_f(t_current, t_current)
                * tr(eps_sw_j) * Identity(3)
            )

        sigma_history += (
            sigma_sw_inst - sigma_sw_relax
        ) * Constant(delta_zeta_j)

    return sigma_history

# ================================================================
# TOTAL VISCOELASTIC STRESS
# ================================================================

def sigma_viscoelastic(
    ue_current,
    Cm_current,
    u_history,
    zeta_history,
    delta_zeta_history,
    t_current,
    t0,
    dt
):

    zeta_current = zeta_history[-1] if len(zeta_history) > 0 else 0.0

    sigma_residual = sigma_residual_only(
        ue_current,
        Cm_current,
        zeta_current
    )

    sigma_history = sigma_history_only(
        u_history,
        zeta_history,
        delta_zeta_history,
        t_current,
        t0,
        dt
    )

    return sigma_residual + sigma_history

# ================================================================
# CARTESIAN TO SPHERICAL STRESS TRANSFORMATION
# ================================================================

def cart_to_spherical_stress_numpy(sig_vals, x):

    r = max(np.sqrt(x[0]**2 + x[1]**2 + x[2]**2), 1e-10)
    rho = max(np.sqrt(x[0]**2 + x[1]**2), 1e-10)

    cos_theta = x[2] / r
    sin_theta = rho / r

    cos_phi = x[0] / rho if rho > 1e-10 else 1.0
    sin_phi = x[1] / rho if rho > 1e-10 else 0.0

    sxx = sig_vals[0]
    syy = sig_vals[4]
    szz = sig_vals[8]

    sxy = sig_vals[1]
    sxz = sig_vals[2]
    syz = sig_vals[5]

    sigma_rr = (
        sxx * sin_theta**2 * cos_phi**2
        + syy * sin_theta**2 * sin_phi**2
        + szz * cos_theta**2
        + 2.0 * sxy * sin_theta**2 * sin_phi * cos_phi
        + 2.0 * sxz * sin_theta * cos_theta * cos_phi
        + 2.0 * syz * sin_theta * cos_theta * sin_phi
    )

    sigma_tt = (
        sxx * cos_theta**2 * cos_phi**2
        + syy * cos_theta**2 * sin_phi**2
        + szz * sin_theta**2
        + 2.0 * sxy * cos_theta**2 * sin_phi * cos_phi
        - 2.0 * sxz * sin_theta * cos_theta * cos_phi
        - 2.0 * syz * sin_theta * cos_theta * sin_phi
    )

    sigma_pp = (
        sxx * sin_phi**2
        + syy * cos_phi**2
        - 2.0 * sxy * sin_phi * cos_phi
    )

    return sigma_rr, sigma_tt, sigma_pp

# ================================================================
# SPHERICAL STRESS EXPRESSIONS
# ================================================================

class SigmaRR(UserExpression):

    def __init__(self, sig, **kwargs):
        super().__init__(**kwargs)
        self.sig = sig

    def eval(self, values, x):
        sig_vals = self.sig(x)
        values[0], _, _ = cart_to_spherical_stress_numpy(sig_vals, x)

    def value_shape(self):
        return ()


class SigmaTT(UserExpression):

    def __init__(self, sig, **kwargs):
        super().__init__(**kwargs)
        self.sig = sig

    def eval(self, values, x):
        sig_vals = self.sig(x)
        _, values[0], _ = cart_to_spherical_stress_numpy(sig_vals, x)

    def value_shape(self):
        return ()


class SigmaPP(UserExpression):

    def __init__(self, sig, **kwargs):
        super().__init__(**kwargs)
        self.sig = sig

    def eval(self, values, x):
        sig_vals = self.sig(x)
        _, _, values[0] = cart_to_spherical_stress_numpy(sig_vals, x)

    def value_shape(self):
        return ()

# ================================================================
# BOUNDARY CONDITIONS
# ================================================================
#
# Boundary ID 1 = symmetry plane
# Boundary ID 2 = fluid-exposed spherical surface
# ================================================================

bc_sym_z = DirichletBC(
    V.sub(3).sub(2),
    Constant(0.0),
    boundaries,
    1
)

bc_fluid = DirichletBC(
    V.sub(0),
    Cf_external,
    boundaries,
    2
)

bcs = [bc_fluid, bc_sym_z]

f_body = Constant((0.0, 0.0, 0.0))

# ================================================================
# HISTORY STORAGE
# ================================================================

u_history = []
zeta_history = []
delta_zeta_history = []

# ================================================================
# TIME
# ================================================================

t0 = 0.0
t = t0

# ================================================================
# NON-MECHANICAL GOVERNING EQUATIONS
# ================================================================

F_fluid = (
    ((Cf - Cf_n) / dt) * v_Cf * dx
    + Df * dot(grad(Cf), grad(v_Cf)) * dx
    + Kf * Cf * v_Cf * dx
)

F_molecular_weight = (
    ((N - N_n) / dt) * v_N * dx
    + k1 * Cf * N * v_N * dx
    + (k2 * N * Cm**n_pow / (1.0 + k3 * Cm**n_pow)) * v_N * dx
)

F_monomer = (
    ((Cm - Cm_n) / dt) * v_Cm * dx
    + Dm * dot(grad(Cm), grad(v_Cm)) * dx
    - (1.0 / N0) * k1 * Cf * N * v_Cm * dx
    - (1.0 / N0)
    * (k2 * N * Cm**n_pow / (1.0 + k3 * Cm**n_pow))
    * v_Cm * dx
)

Fnonmech = F_fluid + F_molecular_weight + F_monomer

# ================================================================
# OUTPUT SPACES
# ================================================================

Vsig = TensorFunctionSpace(mesh, "P", 1)
Vscalar = FunctionSpace(mesh, "P", 1)
Vstrain = TensorFunctionSpace(mesh, "P", 1)

# ================================================================
# TIME LOOP
# ================================================================

i_step = 0

print("\n" + "=" * 70)
print("STARTING HALF-SPHERE LINEAR VISCOELASTIC SIMULATION")
print("=" * 70 + "\n")

while t <= T_final:

    print(f"Step {i_step}, Time = {t:.2f}")

    # Current viscoelastic stress
    sigma_mech = sigma_viscoelastic(
        ue,
        Cm,
        u_history,
        zeta_history,
        delta_zeta_history,
        t,
        t0,
        dt
    )

    Fmech = (
        inner(sigma_mech, epsilon(v_u)) * dx
        - dot(f_body, v_u) * dx
    )

    F = Fnonmech + Fmech

    # Solve coupled problem
    solve(F == 0, u, bcs)

    # Current solution
    Cf_sol, N_sol, Cm_sol, ue_sol = u.split(True)

    # ============================================================
    # UPDATE ZETA HISTORY
    # ============================================================

    domain_volume = assemble(Constant(1.0) * dx)

    N_avg = assemble(N_sol * dx) / domain_volume
    zeta_current = 1.0 - float(N_avg) / float(N0_value)

    if i_step == 0:
        delta_zeta_current = zeta_current
    else:
        delta_zeta_current = zeta_current - zeta_history[-1]

    zeta_history.append(zeta_current)
    delta_zeta_history.append(delta_zeta_current)

    # Store current state for viscoelastic history
    u_history.append(u.copy(deepcopy=True))

    # ============================================================
    # FIELD OUTPUTS
    # ============================================================

    Cf_sol.rename("Conc. of Fluid, Cf", "")
    N_sol.rename("Mol. wt., N", "")
    Cm_sol.rename("Conc. of Monomers, Cm", "")
    ue_sol.rename("Displacement", "")

    results.write(Cf_sol, t)
    results.write(N_sol, t)
    results.write(Cm_sol, t)
    results.write(ue_sol, t)

    # Zeta field
    zeta_field = project(1.0 - N_sol / N0, Vscalar)
    zeta_field.rename("Zeta", "")
    results.write(zeta_field, t)

    # ============================================================
    # AVERAGED VALUES
    # ============================================================

    Cf_avg = assemble(Cf_sol * dx) / domain_volume
    N_avg = assemble(N_sol * dx) / domain_volume
    Cm_avg = assemble(Cm_sol * dx) / domain_volume
    zeta_avg = assemble(zeta_field * dx) / domain_volume

    fluid_surface_area = assemble(Constant(1.0) * ds(2))

    if fluid_surface_area > DOLFIN_EPS:
        Cf_boundary = assemble(Cf_sol * ds(2)) / fluid_surface_area
    else:
        Cf_boundary = 0.0

    # ============================================================
    # STRAIN OUTPUT
    # ============================================================

    eps_proj = project(epsilon(ue_sol), Vstrain)
    eps_proj.rename("Strain", "")
    results.write(eps_proj, t)

    # ============================================================
    # CARTESIAN STRESS OUTPUTS
    # ============================================================

    sigma_total_cart = sigma_viscoelastic(
        ue_sol,
        Cm_sol,
        u_history,
        zeta_history,
        delta_zeta_history,
        t,
        t0,
        dt
    )

    sigma_residual_cart = sigma_residual_only(
        ue_sol,
        Cm_sol,
        zeta_history[-1]
    )

    sigma_history_cart = sigma_history_only(
        u_history,
        zeta_history,
        delta_zeta_history,
        t,
        t0,
        dt
    )

    sig_total_proj = project(sigma_total_cart, Vsig)
    sig_total_proj.rename("Stress_Total_Cart", "MPa")
    results.write(sig_total_proj, t)

    sig_residual_proj = project(sigma_residual_cart, Vsig)
    sig_residual_proj.rename("Stress_Residual_Cart", "MPa")
    results.write(sig_residual_proj, t)

    sig_history_proj = project(sigma_history_cart, Vsig)
    sig_history_proj.rename("Stress_History_Cart", "MPa")
    results.write(sig_history_proj, t)

    # ============================================================
    # SPHERICAL STRESSES - TOTAL
    # ============================================================

    sigma_rr_total = project(SigmaRR(sig_total_proj, degree=1), Vscalar)
    sigma_tt_total = project(SigmaTT(sig_total_proj, degree=1), Vscalar)
    sigma_pp_total = project(SigmaPP(sig_total_proj, degree=1), Vscalar)

    sigma_rr_total.rename("Stress_Total_rr", "MPa")
    sigma_tt_total.rename("Stress_Total_theta", "MPa")
    sigma_pp_total.rename("Stress_Total_phi", "MPa")

    results.write(sigma_rr_total, t)
    results.write(sigma_tt_total, t)
    results.write(sigma_pp_total, t)

    # ============================================================
    # SPHERICAL STRESSES - RESIDUAL
    # ============================================================

    sigma_rr_residual = project(
        SigmaRR(sig_residual_proj, degree=1),
        Vscalar
    )

    sigma_tt_residual = project(
        SigmaTT(sig_residual_proj, degree=1),
        Vscalar
    )

    sigma_pp_residual = project(
        SigmaPP(sig_residual_proj, degree=1),
        Vscalar
    )

    sigma_rr_residual.rename("Stress_Residual_rr", "MPa")
    sigma_tt_residual.rename("Stress_Residual_theta", "MPa")
    sigma_pp_residual.rename("Stress_Residual_phi", "MPa")

    results.write(sigma_rr_residual, t)
    results.write(sigma_tt_residual, t)
    results.write(sigma_pp_residual, t)

    # ============================================================
    # SPHERICAL STRESSES - HISTORY
    # ============================================================

    sigma_rr_history = project(
        SigmaRR(sig_history_proj, degree=1),
        Vscalar
    )

    sigma_tt_history = project(
        SigmaTT(sig_history_proj, degree=1),
        Vscalar
    )

    sigma_pp_history = project(
        SigmaPP(sig_history_proj, degree=1),
        Vscalar
    )

    sigma_rr_history.rename("Stress_History_rr", "MPa")
    sigma_tt_history.rename("Stress_History_theta", "MPa")
    sigma_pp_history.rename("Stress_History_phi", "MPa")

    results.write(sigma_rr_history, t)
    results.write(sigma_tt_history, t)
    results.write(sigma_pp_history, t)

    # ============================================================
    # SAVE AVERAGED VALUES
    # ============================================================

    df.loc[len(df)] = [
        t / 24.0,
        float(Cf_avg),
        float(N_avg),
        float(Cm_avg),
        float(zeta_avg),
        float(Cf_boundary)
    ]

    print(
        f"Time = {t:.2f} hrs, "
        f"Average Zeta = {zeta_current:.6f}, "
        f"Average Cm = {float(Cm_avg):.6f}"
    )

    # Update previous solution
    u_n.assign(u)

    t += dt
    i_step += 1

# ================================================================
# SAVE RESULTS
# ================================================================

df.to_excel(
    os.path.join(directory, "FEA_LVE_HalfSphere_Spherical.xlsx"),
    index=False
)

results.close()

print("\n" + "=" * 70)
print("Simulation completed successfully.")
print("=" * 70)

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