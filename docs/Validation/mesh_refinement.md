# Mesh refinement

## What this page checks

This page checks the mesh that Aragog refines towards the surface and the core-mantle boundary (CMB) when `mesh.surface_cell_thickness` or `mesh.cmb_cell_thickness` is set (see [Mesh refinement](../Explanations/mesh_refinement.md)). Validation here means that we build or run Aragog on a case where we know the right answer and check that Aragog gives it: the end cells must have the thickness that was set, the results must stop changing as the top cell gets thinner, and a mesh without refinement must give exactly the results of the uniform mesh. If this part were wrong, a PROTEUS run with a refined mesh would compute the surface and CMB heat flow over cells of the wrong size, and a run without refinement could change without notice. Each case below says why the check exists, what it tests, how we know the right answer and why it matters, then shows a plot or says why it has none, and the test details.

## The end cells have the thickness that was set

**Why this check exists.** The stretched grid comes from a root solve for the stretching parameters and, in mass-coordinate mode, a second root solve for each node radius. If either solve returns a wrong value, the end cells have the wrong thickness, the nodes cross, or the mantle no longer ends at the CMB and surface radii.

**What it tests.** The mantle of `abe_solid.toml` is 1000 km thick with 100 nodes, so the uniform cell is about 10.1 km in radius mode. The test builds the mesh with a 200 m top cell, a 50 m top cell, and a 200 m top cell together with a 2 km CMB cell, in radius mode and in mass-coordinate mode. For each mesh the node radii must increase, the first and last radius must equal those of the uniform mesh, and each set end cell must equal its target.

**How we know the right answer.** The targets are the values in the config: the test compares the width of each end cell with the number that was set.

**Why it matters.** The surface and CMB heat flow are computed over the end cells, so the end cells decide how well the solver resolves the boundary layers.

![Cell thickness against depth for the uniform mesh and two refined meshes of abe_solid in mass-coordinate mode](../assets/validation/mesh_refinement_end_cells.svg#only-light){ width="80%" }
![Cell thickness against depth for the uniform mesh and two refined meshes of abe_solid in mass-coordinate mode](../assets/validation/mesh_refinement_end_cells_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The x-axis is the depth of each cell centre below the surface, in km; the y-axis is the cell thickness, in m, on a log scale. Each line is one mesh in mass-coordinate mode; the dots mark the two end cells, which the test asserts. The uniform mesh has cells from 9.3 to 10.9 km. With a 50 m top cell the cells grow with depth and reach 38.6 km near the CMB. With a 200 m top cell and a 2 km CMB cell the largest cell is 26.6 km, in the middle of the mantle.

??? info "Test details"
    - Reference: the end-cell thickness set in the config.
    - Tolerance: $10^{-6}$ relative in radius mode; in mass-coordinate mode 2 mm or $10^{-6}$ relative, whichever is larger: the solved node of an end cell is placed to 1 mm, the other is the boundary itself.
    - Test id: `tests/test_mesh_refinement_validation.py::test_the_refined_mesh_has_the_set_end_cells`
    - Command: `pytest tests/test_mesh_refinement_validation.py -m reference_pinned --validation-plots`

## The results converge as the top cell gets thinner

**Why this check exists.** A thinner top cell only helps if the results approach a limit. If they keep changing as the top cell gets thinner, no setting can be recommended.

**What it tests.** Runs of the same magma-ocean case with top cells of 200, 100 and 50 m, at 100 and 200 nodes, with the global melt fraction, the cumulative heat loss, the mean surface heat flux in each time window and the CMB temperature as the compared results.

**How we know the right answer.** The spread between successive top cells must shrink, or fall below the noise floor that a pair of runs with round-off changes to the start state gives.

**Why it matters.** This case sets the recommended top cell for PROTEUS runs.

This case is filled from the convergence study.

## A refined CMB cell and the core

**Why this check exists.** A thinner CMB cell changes the length over which the CMB heat flow is computed, and with it the cooling of the core.

**What it tests.** Two runs at the recommended surface setting with the core module, one with a refined CMB cell and one without; the compared results are the CMB temperature and the CMB heat flow.

**How we know the right answer.** The difference between the two runs is compared with the noise floor of the convergence study.

**Why it matters.** The CMB temperature drives the core evolution in PROTEUS.

This case is filled from the convergence study.

## Without refinement the results equal the uniform mesh

**Why this check exists.** The refinement code sits in the mesh constructor that every run uses. A change to the path without refinement, for example a refined branch that runs when both fields are 0, would change every run without notice.

**What it tests.** With both fields at 0, the test builds the `abe_solid.toml` mesh in mass-coordinate and in radius mode, evaluates the right-hand side once and solves for 100 years with the NumPy right-hand side and with the JAX right-hand side. The mesh arrays and the right-hand side must equal those of the main branch to $10^{-12}$ and the end states to $10^{-9}$, relative to each value or to the largest value of the array. The mesh is not compared exactly because the reference comes from one platform and the round-off of another platform differs: on Linux the radii differ by up to $3 \times 10^{-15}$ relative and the cell volumes by up to $4 \times 10^{-14}$. The test also requires that the JAX solve called the JAX right-hand side, in the stored reference and in the test.

**How we know the right answer.** The reference is the same set of arrays written on the main branch by `tools/verification/generate_mesh_refinement_off_reference.py`.

**Why it matters.** Every run without refinement, the default, must give the same results as before.

This case has no plot: it checks that two sets of arrays are equal, which a plot cannot show.

??? info "Test details"
    - Reference: `tests/reference/mesh_refinement_off_main_reference.npz`, written on the main branch, with its commit and the JAX call count.
    - Tolerance: mesh arrays and right-hand side $10^{-12}$; end states $10^{-9}$.
    - Test id: `tests/test_mesh_refinement_off_reference.py::test_the_refinement_off_path_equals_main`
    - Command: `pytest tests/test_mesh_refinement_off_reference.py`
