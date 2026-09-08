"""
Final 3D density-based SIMP topology-optimisation baseline.

Key features
------------
1. Consistent N-mm-MPa physical unit system for an aluminium cantilever.
2. A cone density-filter radius defined in millimetres.
3. JAX-FEM MMA's internal density and sensitivity filters are explicitly disabled.
4. The actual post-update density returned by MMA is evaluated and saved.
5. Convergence evidence is recorded using:
   - relative compliance change;
   - maximum raw-design-density change;
   - physical-volume error.
6. No passive-solid or passive-void (non-design) regions are imposed.

Important
---------
The convergence summary reports whether the selected numerical tolerances were
actually met. A fixed number of iterations alone is not treated as proof of
convergence.
"""

import csv
import json
import os
import shutil

import jax
import jax.numpy as jnp
import matplotlib
import meshio
import numpy as onp

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from jax import config
from jax_fem.generate_mesh import Mesh, box_mesh
import jax_fem.mma as mma_module
from jax_fem.problem import Problem
from jax_fem.solver import ad_wrapper


config.update("jax_enable_x64", True)


# =============================================================================
# 1. Configuration
# =============================================================================

# Consistent unit system:
#   length       mm
#   force        N
#   stress / E   MPa = N/mm^2
#   compliance   N*mm

NX, NY, NZ = 80, 20, 20
LX, LY, LZ = 160.0, 40.0, 40.0

DX = LX / NX
DY = LY / NY
DZ = LZ / NZ

VOL_TARGET = 0.60

# SIMP material interpolation: provisional aluminium elastic properties.
P_SIMP = 3.0
E_MAX = 70000.0
VOID_STIFFNESS_RATIO = 1.0e-6
E_MIN = VOID_STIFFNESS_RATIO * E_MAX
POISSON_RATIO = 0.33

# MMA settings. The higher maximum iteration count allows convergence to be
# assessed instead of assuming that 70 iterations are sufficient.
NUM_ITERATIONS = 300
MMA_MOVE_LIMIT = 0.03

# The density filter is implemented directly in the differentiable mapping
# rho_design -> rho_physical. Its length scale is specified in millimetres.
USE_DENSITY_FILTER = True
FILTER_RADIUS_MM = 3.0

# Standard uniform SIMP initialisation.
INITIAL_DENSITY = VOL_TARGET

# Idealised benchmark load. The load strip is one element high; this should be
# replaced by a technician-informed load block for the manufacturing case.
TOTAL_APPLIED_LOAD_Y = -100.0
LOAD_STRIP_HEIGHT = LY / NY

# A posteriori convergence criteria.
CONVERGENCE_WINDOW = 10
COMPLIANCE_RELATIVE_TOLERANCE = 1.0e-3
MAX_DESIGN_CHANGE_TOLERANCE = 1.0e-2
VOLUME_ABSOLUTE_TOLERANCE = 1.0e-3

# Saving every iteration can substantially increase runtime because it requires
# an additional forward FE solve. Set to 0 to disable intermediate VTU output.
VTU_SAVE_EVERY = 25

'''
# New directory and file naming for this convergence-checked baseline.
RUN_NAME = "simp_baseline_aluminium_mm_convergence_vf0.6"
OUTPUT_DIR = os.path.join(
    "/home/hy/projects/research_project",
    RUN_NAME,
)

# For a portable relative output directory, use:
# OUTPUT_DIR = os.path.abspath(RUN_NAME)
'''

RUN_NAME = "simp_baseline_aluminium_mm_convergence_vf0.6"

# Repository root.
# This file is stored in <repository>/src/.
REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

# Store generated optimisation results locally under <repository>/outputs/.
OUTPUT_ROOT = os.path.join(
    REPO_ROOT,
    "outputs",
    "simp",
)

OUTPUT_DIR = os.path.join(
    OUTPUT_ROOT,
    RUN_NAME,
)

CLEAR_OLD_RESULTS = True

# =============================================================================
# 2. Disable JAX-FEM MMA's internal filtering
# =============================================================================

# Filtering is already differentiated through get_physical_density(). Applying
# MMA's sensitivity filter afterwards would filter the gradient a second time
# and would no longer represent a standard density-filtered SIMP formulation.
mma_module.density_filtering = False
mma_module.sensitivity_filtering = False


# =============================================================================
# 3. Three-dimensional SIMP elasticity model
# =============================================================================

class DensityElasticity(Problem):

    def __init__(self, *args, **kwargs):
        self.surface_maps = kwargs.pop("surface_maps")
        super().__init__(*args, **kwargs)

    def custom_init(self):
        self.fe = self.fes[0]

        # Every element is a design element. No passive solid or passive void
        # regions are imposed in this idealised baseline.
        self.fe.flex_inds = jnp.arange(len(self.fe.cells))

    def get_tensor_map(self):

        def stress(u_grad, rho):
            rho_value = jnp.clip(rho[0], 0.0, 1.0)

            youngs_modulus = (
                E_MIN
                + (E_MAX - E_MIN) * rho_value**P_SIMP
            )

            mu = youngs_modulus / (
                2.0 * (1.0 + POISSON_RATIO)
            )

            lmbda = (
                youngs_modulus * POISSON_RATIO
                / (
                    (1.0 + POISSON_RATIO)
                    * (1.0 - 2.0 * POISSON_RATIO)
                )
            )

            epsilon = 0.5 * (u_grad + u_grad.T)

            sigma = (
                lmbda * jnp.trace(epsilon) * jnp.eye(3)
                + 2.0 * mu * epsilon
            )

            return sigma

        return stress

    def set_params(self, params):
        """
        Map one physical density per element to all quadrature points.

        Parameters
        ----------
        params
            Array with shape (number_of_elements, 1), or a one-dimensional
            array that can be converted to that shape.
        """

        if params.ndim == 1:
            params = params[:, None]

        full_params = jnp.ones(
            (self.fe.num_cells, params.shape[1]),
            dtype=params.dtype,
        )

        full_params = full_params.at[
            self.fe.flex_inds
        ].set(params)

        thetas = jnp.repeat(
            full_params[:, None, :],
            self.fe.num_quads,
            axis=1,
        )

        self.full_params = full_params
        self.internal_vars = [thetas]

    def get_surface_maps(self):
        return self.surface_maps

    def compute_compliance(self, sol):
        """
        Evaluate C = integral_Gamma(t dot u) dGamma.
        """

        boundary_inds = self.boundary_inds_list[0]

        _, nanson_scale = self.fe.get_face_shape_grads(
            boundary_inds
        )

        u_face_nodes = (
            sol[self.fe.cells][boundary_inds[:, 0]][
                :, None, :, :
            ]
            * self.fe.face_shape_vals[
                boundary_inds[:, 1]
            ][:, :, :, None]
        )

        u_face = jnp.sum(u_face_nodes, axis=2)

        subset_quad_points = (
            self.physical_surface_quad_points[0]
        )

        neumann_fn = self.get_surface_maps()[0]

        # JAX-FEM uses physical traction = -surface_map.
        traction = -jax.vmap(
            jax.vmap(neumann_fn)
        )(u_face, subset_quad_points)

        compliance = jnp.sum(
            traction
            * u_face
            * nanson_scale[:, :, None]
        )

        return compliance


# =============================================================================
# 4. Physical-length three-dimensional density filter
# =============================================================================

def make_density_filter_kernel(
    radius_mm,
    spacing_x,
    spacing_y,
    spacing_z,
):
    """
    Build a physical cone filter:

        w(r) = max(0, radius_mm - r)

    where r is measured in millimetres rather than element indices.
    """

    if radius_mm <= 0.0:
        raise ValueError(
            "FILTER_RADIUS_MM must be strictly positive."
        )

    support_x = int(onp.ceil(radius_mm / spacing_x))
    support_y = int(onp.ceil(radius_mm / spacing_y))
    support_z = int(onp.ceil(radius_mm / spacing_z))

    offsets_x = onp.arange(
        -support_x,
        support_x + 1,
    )
    offsets_y = onp.arange(
        -support_y,
        support_y + 1,
    )
    offsets_z = onp.arange(
        -support_z,
        support_z + 1,
    )

    ix, iy, iz = onp.meshgrid(
        offsets_x,
        offsets_y,
        offsets_z,
        indexing="ij",
    )

    distance_mm = onp.sqrt(
        (ix * spacing_x) ** 2
        + (iy * spacing_y) ** 2
        + (iz * spacing_z) ** 2
    )

    weights = onp.maximum(
        0.0,
        radius_mm - distance_mm,
    )

    if onp.sum(weights) <= 0.0:
        weights[
            support_x,
            support_y,
            support_z,
        ] = 1.0

    kernel = jnp.asarray(
        weights,
        dtype=jnp.float64,
    )

    supports = (
        support_x,
        support_y,
        support_z,
    )

    return kernel, supports


DENSITY_FILTER_KERNEL, FILTER_SUPPORTS = (
    make_density_filter_kernel(
        FILTER_RADIUS_MM,
        DX,
        DY,
        DZ,
    )
)


def density_filter(raw_density_3d):
    """
    Apply the physical cone density filter.

    Zero padding is used for both density and normalisation fields. Dividing by
    the filtered-one field normalises the kernel near domain boundaries and
    preserves a spatially uniform density field.
    """

    if not USE_DENSITY_FILTER:
        return raw_density_3d

    support_x, support_y, support_z = FILTER_SUPPORTS

    padding = (
        (support_x, support_x),
        (support_y, support_y),
        (support_z, support_z),
    )

    padded_density = jnp.pad(
        raw_density_3d,
        padding,
        mode="constant",
        constant_values=0.0,
    )

    padded_ones = jnp.pad(
        jnp.ones_like(raw_density_3d),
        padding,
        mode="constant",
        constant_values=0.0,
    )

    numerator = jax.scipy.signal.convolve(
        padded_density,
        DENSITY_FILTER_KERNEL,
        mode="valid",
    )

    denominator = jax.scipy.signal.convolve(
        padded_ones,
        DENSITY_FILTER_KERNEL,
        mode="valid",
    )

    return numerator / jnp.maximum(
        denominator,
        1.0e-12,
    )


def get_physical_density(rho):
    """
    Convert raw element design variables to physical element densities.

    Input
    -----
    rho
        Shape (number_of_elements,) or (number_of_elements, 1).

    Output
    ------
    rho_physical
        Shape (number_of_elements, 1).
    """

    rho_flat = rho.reshape(-1)
    rho_grid = rho_flat.reshape((NX, NY, NZ))
    rho_filtered = density_filter(rho_grid)

    return jnp.clip(
        rho_filtered,
        0.0,
        1.0,
    ).reshape((-1, 1))


# =============================================================================
# 5. Output and convergence helpers
# =============================================================================

history = []


def save_vtu(
    filename,
    mesh,
    mesh_io,
    rho_design,
    rho_physical,
    displacement,
):
    meshio.write_points_cells(
        filename,
        mesh.points,
        mesh_io.cells,
        point_data={
            "displacement": onp.asarray(
                displacement,
                dtype=float,
            ),
        },
        cell_data={
            "rho_design": [
                onp.asarray(
                    rho_design.reshape(-1),
                    dtype=float,
                )
            ],
            "rho_physical": [
                onp.asarray(
                    rho_physical.reshape(-1),
                    dtype=float,
                )
            ],
        },
    )


def save_density_slices(output_path, rho_cells):
    rho_grid = onp.asarray(
        rho_cells,
        dtype=float,
    ).reshape((NX, NY, NZ))

    x_slice = rho_grid[NX // 2, :, :]
    y_slice = rho_grid[:, NY // 2, :]
    z_slice = rho_grid[:, :, NZ // 2]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 4.8),
    )

    data = [
        (
            x_slice.T,
            "Mid-x slice",
            "y index",
            "z index",
        ),
        (
            y_slice.T,
            "Mid-y slice",
            "x index",
            "z index",
        ),
        (
            z_slice.T,
            "Mid-z slice",
            "x index",
            "y index",
        ),
    ]

    for ax, (
        slice_data,
        title,
        xlabel,
        ylabel,
    ) in zip(axes, data):
        image_plot = ax.imshow(
            slice_data,
            origin="lower",
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
        )

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        colourbar = fig.colorbar(
            image_plot,
            ax=ax,
        )
        colourbar.set_label("Physical density")

    fig.suptitle(
        "Final SIMP physical-density slices",
        fontsize=14,
    )

    fig.tight_layout(
        rect=[0.0, 0.0, 1.0, 0.93]
    )

    fig.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


def write_history_csv(path, history_data):
    if len(history_data) == 0:
        return

    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(history_data[0].keys()),
        )

        writer.writeheader()
        writer.writerows(history_data)


def save_history_plot(
    history_data,
    key,
    ylabel,
    title,
    output_path,
    use_log_scale=False,
):
    if len(history_data) == 0:
        return

    iterations = [
        row["iteration"]
        for row in history_data
    ]

    values = [
        row[key]
        for row in history_data
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iterations, values)

    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if use_log_scale:
        finite_positive = [
            value
            for value in values
            if onp.isfinite(value) and value > 0.0
        ]

        if len(finite_positive) > 0:
            ax.set_yscale("log")

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


def compute_grey_metrics(rho_physical):
    rho_flat = onp.asarray(
        rho_physical,
        dtype=float,
    ).reshape(-1)

    grey_index = float(
        onp.mean(
            4.0 * rho_flat * (1.0 - rho_flat)
        )
    )

    grey_fraction = float(
        onp.mean(
            onp.logical_and(
                rho_flat > 0.1,
                rho_flat < 0.9,
            )
        )
    )

    return grey_index, grey_fraction


def assess_convergence(history_data):
    """
    Assess convergence from the final consecutive updates.

    Convergence requires all of the following:
    1. At least CONVERGENCE_WINDOW finite update records.
    2. Every recent relative compliance change is below its tolerance.
    3. Every recent maximum design change is below its tolerance.
    4. The final physical-volume error is below its tolerance.
    """

    finite_rows = [
        row
        for row in history_data
        if (
            onp.isfinite(
                row["relative_compliance_change"]
            )
            and onp.isfinite(
                row["max_design_change"]
            )
        )
    ]

    enough_history = (
        len(finite_rows) >= CONVERGENCE_WINDOW
    )

    if enough_history:
        recent_rows = finite_rows[
            -CONVERGENCE_WINDOW:
        ]

        recent_relative_changes = [
            abs(
                row["relative_compliance_change"]
            )
            for row in recent_rows
        ]

        recent_design_changes = [
            abs(row["max_design_change"])
            for row in recent_rows
        ]

        maximum_recent_relative_change = float(
            max(recent_relative_changes)
        )

        maximum_recent_design_change = float(
            max(recent_design_changes)
        )
    else:
        maximum_recent_relative_change = float("nan")
        maximum_recent_design_change = float("nan")

    final_volume_error = float(
        abs(history_data[-1]["volume_error"])
    )

    compliance_converged = bool(
        enough_history
        and maximum_recent_relative_change
        <= COMPLIANCE_RELATIVE_TOLERANCE
    )

    design_converged = bool(
        enough_history
        and maximum_recent_design_change
        <= MAX_DESIGN_CHANGE_TOLERANCE
    )

    volume_converged = bool(
        final_volume_error
        <= VOLUME_ABSOLUTE_TOLERANCE
    )

    converged = bool(
        compliance_converged
        and design_converged
        and volume_converged
    )

    return {
        "converged": converged,
        "enough_history": enough_history,
        "window_length": CONVERGENCE_WINDOW,
        "criteria": {
            "maximum_recent_relative_compliance_change":
                COMPLIANCE_RELATIVE_TOLERANCE,
            "maximum_recent_raw_design_change":
                MAX_DESIGN_CHANGE_TOLERANCE,
            "final_absolute_physical_volume_error":
                VOLUME_ABSOLUTE_TOLERANCE,
        },
        "observed": {
            "maximum_recent_relative_compliance_change":
                maximum_recent_relative_change,
            "maximum_recent_raw_design_change":
                maximum_recent_design_change,
            "final_absolute_physical_volume_error":
                final_volume_error,
        },
        "individual_checks": {
            "compliance_converged":
                compliance_converged,
            "design_converged":
                design_converged,
            "volume_converged":
                volume_converged,
        },
    }


# =============================================================================
# 6. Main optimisation
# =============================================================================

def run_simp_baseline():
    global history
    history = []

    # Reset callback state in case this function is called more than once in
    # the same Python process.
    callback_state = {
        "iteration": 0,
        "previous_rho": None,
        "previous_compliance": None,
    }

    if CLEAR_OLD_RESULTS and os.path.exists(
        OUTPUT_DIR
    ):
        shutil.rmtree(OUTPUT_DIR)

    vtu_dir = os.path.join(OUTPUT_DIR, "vtu")
    iteration_vtu_dir = os.path.join(
        vtu_dir,
        "iterations",
    )
    figure_dir = os.path.join(
        OUTPUT_DIR,
        "figures",
    )
    history_dir = os.path.join(
        OUTPUT_DIR,
        "history",
    )
    metadata_dir = os.path.join(
        OUTPUT_DIR,
        "metadata",
    )

    for directory in (
        iteration_vtu_dir,
        figure_dir,
        history_dir,
        metadata_dir,
    ):
        os.makedirs(directory, exist_ok=True)

    print(
        "Saving convergence-checked SIMP baseline to:",
        OUTPUT_DIR,
        flush=True,
    )

    # -------------------------------------------------------------------------
    # Mesh
    # -------------------------------------------------------------------------

    mesh_io = box_mesh(
        NX,
        NY,
        NZ,
        LX,
        LY,
        LZ,
    )

    mesh = Mesh(
        mesh_io.points,
        mesh_io.cells_dict["hexahedron"],
        ele_type="HEX8",
    )

    # -------------------------------------------------------------------------
    # Boundary conditions
    # -------------------------------------------------------------------------

    def loc_clamp(x):
        return jnp.isclose(
            x[0],
            0.0,
            atol=1.0e-5,
        )

    def zero_displacement(x):
        del x
        return 0.0

    dirichlet_bc = [
        [
            loc_clamp,
            loc_clamp,
            loc_clamp,
        ],
        [0, 1, 2],
        [
            zero_displacement,
            zero_displacement,
            zero_displacement,
        ],
    ]

    def loc_load(x):
        return jnp.logical_and(
            jnp.isclose(
                x[0],
                LX,
                atol=1.0e-5,
            ),
            x[1]
            <= LOAD_STRIP_HEIGHT + 1.0e-5,
        )

    def surface_map(u, x):
        del u, x

        loaded_area = (
            LOAD_STRIP_HEIGHT * LZ
        )

        return jnp.array(
            [
                0.0,
                -TOTAL_APPLIED_LOAD_Y
                / loaded_area,
                0.0,
            ]
        )

    # -------------------------------------------------------------------------
    # Forward and adjoint FE problem
    # -------------------------------------------------------------------------

    problem = DensityElasticity(
        mesh,
        vec=3,
        dim=3,
        ele_type="HEX8",
        dirichlet_bc_info=dirichlet_bc,
        location_fns=[loc_load],
        surface_maps=[surface_map],
    )

    fwd_pred = ad_wrapper(
        problem,
        solver_options={
            "spsolve_solver": {}
        },
        adjoint_solver_options={
            "spsolve_solver": {}
        },
    )

    # -------------------------------------------------------------------------
    # Objective and constraint
    # -------------------------------------------------------------------------

    def J_total(rho):
        rho_physical = get_physical_density(rho)
        displacement = fwd_pred(
            rho_physical
        )[0]

        return problem.compute_compliance(
            displacement
        )

    def volume_constraint(rho):
        rho_physical = get_physical_density(rho)

        return (
            jnp.mean(rho_physical)
            / VOL_TARGET
            - 1.0
        )

    # -------------------------------------------------------------------------
    # Iteration recording callback
    # -------------------------------------------------------------------------

    def record_state(
        rho,
        compliance,
        is_final=False,
        displacement=None,
    ):
        rho_jnp = jnp.asarray(rho)
        rho_physical = get_physical_density(
            rho_jnp
        )

        rho_design_np = onp.asarray(
            rho_jnp,
            dtype=float,
        ).reshape(-1)

        rho_physical_np = onp.asarray(
            rho_physical,
            dtype=float,
        ).reshape(-1)

        compliance_value = float(compliance)
        volume_fraction = float(
            onp.mean(rho_physical_np)
        )

        if callback_state["previous_rho"] is None:
            max_design_change = float("nan")
        else:
            max_design_change = float(
                onp.max(
                    onp.abs(
                        rho_design_np
                        - callback_state[
                            "previous_rho"
                        ]
                    )
                )
            )

        previous_compliance = callback_state[
            "previous_compliance"
        ]

        if previous_compliance is None:
            relative_compliance_change = float("nan")
        else:
            relative_compliance_change = float(
                abs(
                    compliance_value
                    - previous_compliance
                )
                / max(
                    abs(previous_compliance),
                    1.0e-16,
                )
            )

        iteration = callback_state["iteration"]

        history.append(
            {
                "iteration": iteration,
                "compliance_N_mm":
                    compliance_value,
                "compliance":
                    compliance_value,
                "volume_fraction":
                    volume_fraction,
                "volume_error":
                    volume_fraction
                    - VOL_TARGET,
                "relative_compliance_change":
                    relative_compliance_change,
                "max_design_change":
                    max_design_change,
                "is_final":
                    bool(is_final),
            }
        )

        should_save_iteration = bool(
            not is_final
            and VTU_SAVE_EVERY > 0
            and iteration % VTU_SAVE_EVERY == 0
        )

        if should_save_iteration:
            if displacement is None:
                displacement = fwd_pred(
                    rho_physical
                )[0]

            save_vtu(
                os.path.join(
                    iteration_vtu_dir,
                    (
                        "simp_baseline_aluminium_"
                        f"iter_{iteration:04d}.vtu"
                    ),
                ),
                mesh,
                mesh_io,
                rho_design_np,
                rho_physical_np,
                displacement,
            )

        print(
            f"Iter {iteration:03d} | "
            f"C: {compliance_value:.8e} N*mm | "
            f"Vol: {volume_fraction:.6f} | "
            f"VolErr: "
            f"{volume_fraction - VOL_TARGET:+.3e} | "
            f"RelC: "
            f"{relative_compliance_change:.3e} | "
            f"MaxDrho: {max_design_change:.3e}"
            + (" | FINAL" if is_final else ""),
            flush=True,
        )

        callback_state["previous_rho"] = (
            rho_design_np.copy()
        )
        callback_state["previous_compliance"] = (
            compliance_value
        )
        callback_state["iteration"] += 1

    def objective_handle(rho):
        objective, gradient = jax.value_and_grad(
            J_total
        )(rho)

        record_state(
            rho,
            objective,
            is_final=False,
        )

        return objective, gradient

    def constraint_handle(rho, epoch):
        del epoch

        constraint, gradient = jax.value_and_grad(
            volume_constraint
        )(rho)

        return (
            constraint.reshape((1,)),
            gradient[None, ...],
        )

    # -------------------------------------------------------------------------
    # Initial design
    # -------------------------------------------------------------------------

    number_of_elements = len(
        problem.fe.flex_inds
    )

    rho_initial = (
        INITIAL_DENSITY
        * jnp.ones(
            (number_of_elements, 1),
            dtype=jnp.float64,
        )
    )

    initial_physical_volume = float(
        jnp.mean(
            get_physical_density(
                rho_initial
            )
        )
    )

    print(
        "Settings | "
        f"mesh = {NX} x {NY} x {NZ} | "
        f"element = "
        f"{DX:.3f} x {DY:.3f} x {DZ:.3f} mm | "
        f"filter radius = "
        f"{FILTER_RADIUS_MM:.3f} mm | "
        f"initial volume = "
        f"{initial_physical_volume:.6f} | "
        f"target volume = {VOL_TARGET:.6f}",
        flush=True,
    )

    print(
        "JAX-FEM MMA internal filters | "
        f"density = "
        f"{mma_module.density_filtering} | "
        f"sensitivity = "
        f"{mma_module.sensitivity_filtering}",
        flush=True,
    )

    # -------------------------------------------------------------------------
    # Run MMA and retain its actual post-update final design
    # -------------------------------------------------------------------------

    optimisation_parameters = {
        "maxIters": NUM_ITERATIONS,
        "movelimit": MMA_MOVE_LIMIT,
    }

    rho_final = mma_module.optimize(
        problem.fe,
        rho_initial,
        optimisation_parameters,
        objective_handle,
        constraint_handle,
        numConstraints=1,
    )

    rho_final = jnp.asarray(
        rho_final,
        dtype=jnp.float64,
    )

    # This is a fresh evaluation of the actual density returned after the last
    # MMA update. It is not the pre-update state from the final callback.
    rho_physical_final = get_physical_density(
        rho_final
    )

    displacement_final = fwd_pred(
        rho_physical_final
    )[0]

    compliance_final = (
        problem.compute_compliance(
            displacement_final
        )
    )

    record_state(
        rho_final,
        compliance_final,
        is_final=True,
        displacement=displacement_final,
    )

    # -------------------------------------------------------------------------
    # Save final result
    # -------------------------------------------------------------------------

    rho_final_np = onp.asarray(
        rho_final,
        dtype=float,
    )

    rho_physical_final_np = onp.asarray(
        rho_physical_final,
        dtype=float,
    )

    displacement_final_np = onp.asarray(
        displacement_final,
        dtype=float,
    )

    final_vtu_path = os.path.join(
        vtu_dir,
        "simp_baseline_aluminium_final.vtu",
    )

    save_vtu(
        final_vtu_path,
        mesh,
        mesh_io,
        rho_final_np,
        rho_physical_final_np,
        displacement_final_np,
    )

    onp.savez(
        os.path.join(
            OUTPUT_DIR,
            "simp_baseline_aluminium_final_fields.npz",
        ),
        rho_design=rho_final_np,
        rho_physical=rho_physical_final_np,
        displacement=displacement_final_np,
    )

    save_density_slices(
        os.path.join(
            figure_dir,
            "simp_baseline_aluminium_final_slices.png",
        ),
        rho_physical_final_np,
    )

    # -------------------------------------------------------------------------
    # Save convergence history and evidence
    # -------------------------------------------------------------------------

    history_csv_path = os.path.join(
        history_dir,
        "simp_baseline_aluminium_history.csv",
    )

    write_history_csv(
        history_csv_path,
        history,
    )

    save_history_plot(
        history,
        "compliance",
        "Compliance (N mm)",
        "SIMP compliance convergence",
        os.path.join(
            figure_dir,
            "simp_baseline_aluminium_compliance.png",
        ),
    )

    save_history_plot(
        history,
        "volume_fraction",
        "Physical volume fraction",
        "SIMP physical-volume convergence",
        os.path.join(
            figure_dir,
            "simp_baseline_aluminium_volume.png",
        ),
    )

    save_history_plot(
        history,
        "relative_compliance_change",
        "Relative compliance change",
        "SIMP relative compliance change",
        os.path.join(
            figure_dir,
            (
                "simp_baseline_aluminium_"
                "relative_compliance_change.png"
            ),
        ),
        use_log_scale=True,
    )

    save_history_plot(
        history,
        "max_design_change",
        "Maximum raw-density change",
        "SIMP maximum design change",
        os.path.join(
            figure_dir,
            (
                "simp_baseline_aluminium_"
                "maximum_design_change.png"
            ),
        ),
        use_log_scale=True,
    )

    convergence_summary = assess_convergence(
        history
    )

    final_volume_fraction = float(
        jnp.mean(rho_physical_final)
    )

    maximum_displacement = float(
        onp.max(
            onp.linalg.norm(
                displacement_final_np,
                axis=1,
            )
        )
    )

    grey_index, grey_fraction = (
        compute_grey_metrics(
            rho_physical_final_np
        )
    )

    convergence_summary[
        "final_result"
    ] = {
        "compliance_N_mm":
            float(compliance_final),
        "physical_volume_fraction":
            final_volume_fraction,
        "physical_volume_error":
            final_volume_fraction
            - VOL_TARGET,
        "maximum_displacement_mm":
            maximum_displacement,
        "grey_index":
            grey_index,
        "grey_fraction_0p1_to_0p9":
            grey_fraction,
        "final_vtu":
            final_vtu_path,
    }

    with open(
        os.path.join(
            metadata_dir,
            "simp_baseline_aluminium_convergence.json",
        ),
        "w",
    ) as file:
        json.dump(
            convergence_summary,
            file,
            indent=2,
        )

    settings = {
        "run_name": RUN_NAME,
        "unit_system": "N-mm-MPa",
        "mesh": {
            "NX": NX,
            "NY": NY,
            "NZ": NZ,
            "LX_mm": LX,
            "LY_mm": LY,
            "LZ_mm": LZ,
            "DX_mm": DX,
            "DY_mm": DY,
            "DZ_mm": DZ,
        },
        "material": {
            "E_MAX_MPa": E_MAX,
            "void_stiffness_ratio":
                VOID_STIFFNESS_RATIO,
            "E_MIN_MPa": E_MIN,
            "poisson_ratio":
                POISSON_RATIO,
            "SIMP_penalty": P_SIMP,
        },
        "optimisation": {
            "volume_target": VOL_TARGET,
            "initial_density":
                INITIAL_DENSITY,
            "maximum_iterations":
                NUM_ITERATIONS,
            "MMA_move_limit":
                MMA_MOVE_LIMIT,
            "internal_density_filtering":
                bool(
                    mma_module.density_filtering
                ),
            "internal_sensitivity_filtering":
                bool(
                    mma_module.sensitivity_filtering
                ),
        },
        "filter": {
            "enabled": USE_DENSITY_FILTER,
            "type": "physical 3D cone",
            "radius_mm": FILTER_RADIUS_MM,
            "kernel_shape": list(
                DENSITY_FILTER_KERNEL.shape
            ),
        },
        "load": {
            "total_y_load_N":
                TOTAL_APPLIED_LOAD_Y,
            "load_strip_height_mm":
                LOAD_STRIP_HEIGHT,
        },
        "non_design_regions": False,
        "convergence": {
            "window":
                CONVERGENCE_WINDOW,
            "relative_compliance_tolerance":
                COMPLIANCE_RELATIVE_TOLERANCE,
            "maximum_design_change_tolerance":
                MAX_DESIGN_CHANGE_TOLERANCE,
            "absolute_volume_error_tolerance":
                VOLUME_ABSOLUTE_TOLERANCE,
        },
    }

    with open(
        os.path.join(
            metadata_dir,
            "simp_baseline_aluminium_settings.json",
        ),
        "w",
    ) as file:
        json.dump(
            settings,
            file,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # Final terminal summary
    # -------------------------------------------------------------------------

    print(
        "\nSIMP optimisation finished.",
        flush=True,
    )

    print(
        f"Final compliance: "
        f"{float(compliance_final):.8e} N*mm",
        flush=True,
    )

    print(
        f"Final physical volume fraction: "
        f"{final_volume_fraction:.8f}",
        flush=True,
    )

    print(
        f"Final maximum displacement: "
        f"{maximum_displacement:.8e} mm",
        flush=True,
    )

    print(
        f"Grey index: {grey_index:.8e}",
        flush=True,
    )

    print(
        f"Grey fraction (0.1 < rho < 0.9): "
        f"{grey_fraction:.8e}",
        flush=True,
    )

    print(
        "Convergence criteria satisfied:",
        convergence_summary["converged"],
        flush=True,
    )

    print(
        "Convergence details:",
        convergence_summary["observed"],
        flush=True,
    )

    if not convergence_summary["converged"]:
        print(
            "WARNING: The selected convergence criteria were not all met. "
            "Do not report this run as converged. Increase the iteration "
            "limit or investigate the failing criterion before using the "
            "result as the final thesis baseline.",
            flush=True,
        )

    print(
        "Final VTU:",
        final_vtu_path,
        flush=True,
    )

    print(
        "History CSV:",
        history_csv_path,
        flush=True,
    )

    print(
        "All results saved in:",
        OUTPUT_DIR,
        flush=True,
    )


if __name__ == "__main__":
    run_simp_baseline()
