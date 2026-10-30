# Mesh refinement

## What this page checks

This page checks the mesh that Aragog refines towards the surface and the core-mantle boundary (CMB) when `mesh.surface_cell_thickness` or `mesh.cmb_cell_thickness` is set (see [Mesh refinement](../Explanations/mesh_refinement.md)). Validation here means that we build or run Aragog on a case where we know the right answer and check that Aragog gives it: the end cells must have the thickness that was set, the results must stop changing as the top cell gets thinner, and a mesh without refinement must give the results of the main branch to the round-off of another platform. If this part were wrong, a PROTEUS run with a refined mesh would compute the surface and CMB heat flow over cells of the wrong size, and a run without refinement could change without notice. Each case below says why the check exists, what it tests, how we know the right answer and why it matters, then shows a plot or says why it has none, and the test details.

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

**What it tests.** The case is the magma ocean of `abe_solid.toml` with a molten start (entropy 9466 J/kg/K), the stagnant-lid rheology and the PALEOS MgSiO$_3$ tables with the low-temperature extension (`paleos_lowt_spider_eos_1350x281`). It runs to $10^5$ years with top cells of 200, 100 and 50 m, at 100 and at 200 nodes. The CMB cell is held at the uniform CMB cell of each node count, 9327 m and 4638 m, so that only the top cell changes. The compared results are the global melt fraction at $10^3$, $10^4$ and $10^5$ years, the heat lost through the surface up to $10^5$ years, the mean surface heat flux in the windows $10^2$ to $10^3$, $10^3$ to $10^4$ and $10^4$ to $10^5$ years, and the CMB temperature at $10^3$, $10^4$ and $10^5$ years: 10 results per node count.

**How we know the right answer.** There is no exact solution, so the test asks whether the results settle. For each result $X$ it takes the spread between the 200 m and the 100 m run, $s_\mathrm{c} = |X_{200}/X_{100} - 1|$, and between the 100 m and the 50 m run, $s_\mathrm{f} = |X_{100}/X_{50} - 1|$. For an error that falls in proportion to the top cell, halving the cell halves the spread, so $s_\mathrm{f}/s_\mathrm{c} = 0.5$. A result passes when $s_\mathrm{f}/s_\mathrm{c} \le 0.75$ or when $s_\mathrm{f}$ is below the noise floor. The noise floor $e$ comes from two more 50 m runs whose start entropy differs from the first by $\pm 10^{-12}$ relative: $e = |X_+/X_- - 1|$, and no difference below $3e$ is counted. A result with $s_\mathrm{c} \le 3e$ counts as converged at 200 m only when $s_\mathrm{f} \le 3e$ too; otherwise it does not pass.

**Why it matters.** This case sets the recommended top cell for PROTEUS runs.

![Change of four results against the top cell thickness at 100 and 200 nodes](../assets/validation/mesh_refinement_convergence.svg#only-light){ width="90%" }
![Change of four results against the top cell thickness at 100 and 200 nodes](../assets/validation/mesh_refinement_convergence_dark.svg#only-dark){ width="90%" }

**How to read the plot.** Each panel shows one result as its change, in percent, from the 50 m run at 200 nodes, against the top cell thickness on a log axis: the global melt fraction at $10^5$ years (top left), the heat lost up to $10^5$ years (top right), the mean surface heat flux from $10^4$ to $10^5$ years (bottom left) and the CMB temperature at $10^5$ years (bottom right). Circles are runs at 100 nodes, squares runs at 200 nodes. The grey band is the $\pm 1$ % tolerance of the recommendation; the noise floor $3e$ is at most 0.04 % for these four results, too narrow to draw. A result that converges in the top cell moves towards 0 from right to left. The gap between the circles and the squares at the same top cell is the effect of the node count: for the melt fraction it is larger than the change between top cells.

**What the study finds.** At 200 nodes 5 of the 10 results pass: the melt fraction at $10^3$ and $10^4$ years, the CMB temperature at $10^3$ years, the heat lost up to $10^5$ years and the mean surface heat flux from $10^4$ to $10^5$ years. At 100 nodes 5 of the 10 pass. The noise floor $3e$ lies between $7.4 \times 10^{-8}$ and $1.4 \times 10^{-3}$ at 200 nodes. The CMB temperature of the 200 m and 100 m runs differs from the 50 m run by at most $3.4 \times 10^{-5}$, but its spreads do not shrink at $10^4$ and $10^5$ years.

- **The recommended top cell is 50 m.** The rule takes, for each result at 200 nodes, the largest top cell that stays within $\max(3e, 1\,\%)$ of the 50 m run, together with every thinner top cell. The melt fraction at all three times and the mean surface heat flux after $10^3$ years need 50 m: with a 100 m top cell the melt fraction at $10^5$ years differs by 2.2 % and the mean surface heat flux from $10^4$ to $10^5$ years by 1.7 %.
- **100 nodes are not enough.** At a 50 m top cell, going from 100 to 200 nodes changes the CMB temperature at $10^5$ years by $4.0 \times 10^{-4}$, the heat lost up to $10^5$ years by $3.5 \times 10^{-3}$ and the melt fraction at $10^5$ years by 5.1 %. These changes are larger than the change from a 100 m to a 50 m top cell for the same results ($3.2 \times 10^{-5}$, $5.8 \times 10^{-4}$ and 2.2 % at 200 nodes). The mean surface heat flux from $10^4$ to $10^5$ years changes by 2.7 % with the node count and by 1.7 % with the top cell.
- **The melt fraction and the heat loss at a fixed time sample one phase of a cycle.** In every run the top cell freezes for the first time between 768 and 791 years. After that it freezes and melts again in bursts, and the global melt fraction falls in steps and recovers slowly between them. A value at one time depends on where in the cycle that time falls, so the verdicts for these results are limited by the phase. The time mean of the melt fraction over the second half of each interval is less sensitive to the phase: at 200 nodes it passes at $10^3$ and $10^4$ years and not at $10^5$ years. This mean was defined after the runs and is not part of the pass rule. The CMB temperature does not cycle.
- **The first frozen interval follows the top cell.** It lasts 2851 years on the uniform 100-node mesh (top cell 10.9 km), 85 years with a 200 m top cell and 64 years with a 50 m top cell at 100 nodes, and 51 years at 50 m and 200 nodes. It does not approach a limit, so it is reported and not tested for convergence.
- **Cost.** Up to $10^5$ years the solver takes $2.4 \times 10^5$ steps on the uniform 100-node mesh, $1.4 \times 10^6$ with a 200 m top cell and $1.9 \times 10^6$ with a 50 m top cell, both at 100 nodes.

**The long run.** A 50 m top cell at 100 nodes runs to $10^8$ years. Its first $10^5$ years equal the 50 m run above. The global melt fraction is 0.065 at $10^6$ years, $3.3 \times 10^{-5}$ at $10^7$ years and $2.1 \times 10^{-5}$ at $10^8$ years, and the mean surface heat flux falls from 163 W/m$^2$ ($10^5$ to $10^6$ years) to 0.096 W/m$^2$ ($10^7$ to $10^8$ years). The CMB temperature is 3465.0 K at $10^6$ years and 3440.3 K at $10^7$ and $10^8$ years.

**Why the CMB temperature stops at 3440.3 K.**

- *Why it happens.* The solid temperature table has valid entries only where PALEOS gives a value for every property. At the CMB pressure, 43.38 GPa, the last valid entropy node is 3070.70 J/kg/K, below the solidus entropy of the lowest cell. At the next node, 3150.82 J/kg/K, and above it, the table repeats the temperature of the last valid node: 3439.49 K and 3443.24 K at the two pressure nodes around the CMB pressure.
- *What it is.* The entropy of the lowest cell is 3111.09 J/kg/K at $10^7$ years and 3110.24 J/kg/K at $10^8$ years. Both values lie between the last valid node and the next one. Both nodes give the same temperature, so the table gives 3440.3 K for every entropy in between. The CMB temperature stays constant while the entropy still falls.
- *Why it matters.* After $10^7$ years the CMB temperature of this run is set by the table, not by the cooling of the mantle. A run that uses it, for example one coupled to the core, needs a table with valid entries up to the solidus at the CMB.

**The uniform mesh in the long run.** The uniform 100-node mesh (top cell 10.9 km) also runs to $10^8$ years with the same settings. Its first $10^5$ years equal the uniform run above. At $10^7$ and $10^8$ years its melt fraction is more than $10^3$ times that of the 50 m mesh:

| Result | Mesh | $10^6$ years | $10^7$ years | $10^8$ years |
|---|---|---|---|---|
| Global melt fraction | uniform | 0.12 | 0.12 | 0.079 |
| | 50 m top cell | 0.065 | $3.3 \times 10^{-5}$ | $2.1 \times 10^{-5}$ |
| CMB temperature [K] | uniform | 3495.6 | 3493.1 | 3463.3 |
| | 50 m top cell | 3465.0 | 3440.3 | 3440.3 |
| Mean surface heat flux [W/m$^2$] | uniform | 55 | 0.61 | 0.71 |
| | 50 m top cell | 163 | 8.0 | 0.096 |

The mean surface heat flux is the mean over the decade that ends at the given time.

**An alternating pattern on the 50 m mesh.** On the 50 m mesh the difference between the entropy and the solidus entropy changes sign from one cell to the next. In the lowest 48 cells it changes sign 0 times at $10^6$ years, 39 times at $10^7$ years and 10 times at $10^8$ years, out of 47 cell pairs. At $10^7$ years 53 of the 99 cells lie within 5 J/kg/K of the solidus, and 38 at $10^8$ years. The uniform mesh does not show the pattern, but it cannot test it: none of its cells comes within 145 J/kg/K of the solidus up to $10^8$ years. The cause of the pattern is not known.

**Limits of the tables.** The PALEOS liquid table in use agrees with the MgSiO$_3$ liquid table of the PALEOS release on Zenodo ([10.5281/zenodo.19680050](https://doi.org/10.5281/zenodo.19680050)) at 1 bar. On all 122 valid melt nodes at 1 bar, from 1831 K to 4862 K, the density differs by at most $3.4 \times 10^{-5}$ (relative) and the entropy by at most 0.34 J/kg/K, with the Zenodo values interpolated linearly in temperature. That release has a known density step on the liquid branch, in which the entropy falls as the temperature rises ([PALEOS issue 7](https://github.com/maraattia/PALEOS/issues/7)). At 1 bar the step lies between 1595 K and 1620 K, where the density rises from 2227 to 3863 kg/m$^3$, below the lowest valid melt node. The study uses the table as it is and was not repeated with a corrected one.

**Not checked.**

- How much the phase of the cycle changes the mean surface heat flux from $10^4$ to $10^5$ years and the heat lost up to $10^5$ years.
- Whether 400 nodes change the interior results less than the step from 100 to 200 nodes.
- How the cooling of the long run changes when the table has valid entries up to the solidus at the CMB.
- Whether the alternating pattern follows the cell index (a 50 m top cell at 200 nodes to $10^8$ years) or comes from round-off (the 50 m mesh from a perturbed start).

??? info "Test details"
    - Reference: the raw values of the study runs in `tests/validation/data/mesh_refinement_study.json`, written by `tools/verification/run_mesh_refinement_study.py` from the run folders; the test computes the noise floor, the spreads, the verdicts, the recommendation and every number above from them.
    - Runs: aragog commit bbef9c9f on Habrok, PALEOS table `temperature_solid.dat` md5 18d6f9eb5ba654d58fe3fad30ce118f0.
    - Tolerance: each quoted number to the digits it is quoted with.
    - Tables: the solid table at the CMB pressure and the melt table at 1 bar, with their md5, and the 1 bar rows of `paleos_mgsio3_tables_pt_proteus_liquid.dat` from Zenodo 19680050 (md5 4b5c14255ca845880a2e940110861df1), in the header of the same file.
    - Test ids: `tests/test_mesh_refinement_study.py::test_the_results_converge_as_the_top_cell_gets_thinner` and `tests/test_mesh_refinement_study.py::test_the_long_run_and_the_tables`
    - Command: `pytest tests/test_mesh_refinement_study.py -m reference_pinned --validation-plots`

## A refined CMB cell and the core

**Why this check exists.** A thinner CMB cell changes the length over which the CMB heat flow is computed, and with it the cooling of the core.

**What it tests.** Two runs at the recommended surface setting with the core module, one with a refined CMB cell and one without; the compared results are the CMB temperature and the CMB heat flow.

**How we know the right answer.** The difference between the two runs is compared with the noise floor of the convergence study.

**Why it matters.** The CMB temperature drives the core evolution in PROTEUS.

This case is filled from the convergence study.

## Without refinement the results equal the main branch

**Why this check exists.** The refinement code sits in the mesh constructor that every run uses. A change to the path without refinement, for example a refined branch that runs when both fields are 0, would change every run without notice.

**What it tests.** With both fields at 0, the test builds the `abe_solid.toml` mesh in mass-coordinate and in radius mode, evaluates the right-hand side once and solves for 100 years with the NumPy right-hand side and with the JAX right-hand side. The mesh arrays and the right-hand side must equal those of the main branch to $10^{-10}$ and the end states to $10^{-6}$, relative to each value or to the largest value of the array. The arrays are not compared exactly because the reference comes from macOS and the round-off of Linux differs: there the largest differences are $1.0 \times 10^{-12}$ of the largest value for the mesh arrays, $2.8 \times 10^{-14}$ for the right-hand side and $2.3 \times 10^{-8}$ for the end states. The mesh arrays and the right-hand side have the tight tolerance; the end-state comparison is a coarse check. In radius mode a refined branch run with both fields at 0 gives the uniform mesh to round-off, which no tolerance that covers the round-off of another platform can tell apart. So a unit test checks that the mesh with both fields at 0 does not call the refined grid, its radii check or, in radius mode, the cell-mass density, and that it solves every mass-coordinate radius to 1 m. The test also requires that the JAX solve called the JAX right-hand side, in the stored reference and in the test.

**How we know the right answer.** The reference is the same set of arrays written on the main branch by `tools/verification/generate_mesh_refinement_off_reference.py`.

**Why it matters.** Every run without refinement, the default, must give the same results as before.

This case has no plot: it checks that two sets of arrays are equal, which a plot cannot show.

??? info "Test details"
    - Reference: `tests/reference/mesh_refinement_off_main_reference.npz`, written on the main branch, with its commit and the JAX call count.
    - Tolerance: mesh arrays and right-hand side $10^{-10}$; end states $10^{-6}$.
    - Test id: `tests/test_mesh_refinement_off_reference.py::test_the_refinement_off_path_equals_main`
    - Command: `pytest tests/test_mesh_refinement_off_reference.py`
