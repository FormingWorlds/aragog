# Mesh refinement towards the boundaries

How can Aragog use thin cells at the surface and at the core-mantle boundary (CMB) without more nodes?

Aragog keeps the number of nodes and moves them: the cells grow smoothly from a thin end cell towards the middle of the mantle, so a thin end cell costs no extra memory. By default the CMB cell is 3000 m thick and the top cell is as thick as on the uniform mesh ([The default mesh](#the-default-mesh)); the core module takes the heat flow out of the core across the CMB cell. You can set the thickness of the outermost cell, the innermost cell, or both, in metres. With `cmb_cell_thickness = 0` and the top cell `"uniform"` (or 0), the mesh is the uniform mesh.

## Why the end cells matter

The heat flow through the surface and through the CMB is computed over the end cells. The core module takes the CMB heat flux across the lower half of the first cell ([Core boundary conditions](core_bc.md)), and a prescribed surface temperature acts over half the top cell ([Surface boundary conditions](surface_bc.md)). The end cells therefore set the length over which the solver resolves the steep temperature changes at the two boundaries. On a mesh uniform in radius the end cells are as thick as all other cells: 10.1 km for the 1000 km mantle of `abe_solid.toml` with 100 nodes. In the default mass-coordinate mode the uniform CMB cell is 9.33 km and the uniform top cell 10.9 km.

## The stretched grid

The mesh starts from a computational coordinate $\xi$, the node index scaled to run uniformly from 0 to 1. A stretching function $s(\xi)$, also from 0 to 1, gives the node positions. The node radii are $r = r_\mathrm{c} + (r_\mathrm{s} - r_\mathrm{c})\, s(\xi)$, with $r_\mathrm{c}$ the CMB radius and $r_\mathrm{s}$ the surface radius, both in m. A uniform mesh has $s = \xi$.

With a target only at the surface, Aragog uses the one-sided hyperbolic tangent

$$
s(\xi) = \frac{\tanh(\beta \xi)}{\tanh \beta},
$$

where $\beta > 0$ is a dimensionless stretching parameter. A larger $\beta$ puts the nodes closer together near $\xi = 1$. Aragog solves for the $\beta$ that makes the last cell, $1 - s(1 - h)$ with $h = 1/(n - 1)$ and $n$ the number of nodes, equal to the target fraction. A target only at the CMB uses the mirror image of the same map.

With targets at both ends, Aragog uses the two-sided function of Vinokur (1983)[^cite-vinokur1983],

$$
s(\xi) = \frac{u}{A + (1 - A)\, u}, \qquad u = \frac{1}{2} \left( 1 + \frac{\tanh\left(\delta (\xi - 1/2)\right)}{\tanh(\delta / 2)} \right),
$$

with two dimensionless parameters: $\delta$ sets how strongly the nodes cluster at both ends, and $A$ sets the balance between the two ends. Aragog solves for $\delta$ and $A$ so that the first and the last cell both equal their targets, and refuses the grid when the solve does not converge or the nodes do not increase.

## Mass-coordinate mode

With `mass_coordinates = true` (the default), the uniform grid is uniform in the mass coordinate $\xi_m$, not in radius ([Model overview](model.md)). The stretching then acts on $\xi_m$, and Aragog converts each target from metres into a mass-coordinate fraction, so the end cells still have the thickness you set, in radius. On a refined mesh the basic node radii come from the mass-coordinate solve, to 1 mm; the staggered node radii keep the 1 m tolerance of the uniform mesh, which moves them by less than $10^{-4}$ of their cell.

## Worked example

`abe_solid.toml` has a 1000 km mantle and 100 nodes, so the uniform cell in radius mode is 10.1 km. In radius mode:

- A 1 km top cell makes the bottom cell grow to 23.4 km. The larger of two neighbouring cells is at most 4.6 % larger than the smaller.
- A 1 km top cell and a 2 km CMB cell make the largest cell 21.5 km. The larger of two neighbouring cells is at most 8.4 % larger than the smaller.

In mass-coordinate mode the same settings give a bottom cell of 22.1 km in the first case and a largest cell of 21.5 km in the second.

## The default mesh

Without the two fields the CMB cell is 3000 m and the top cell is `"uniform"`: the top cell of the uniform mesh of the same coordinate mode, $(r_\mathrm{s} - r_\mathrm{c})/(n - 1)$ in radius mode and the top cell of the uniform mass-coordinate mesh in mass-coordinate mode. The stretched grid then holds both ends, so the cells grow from the CMB towards the middle of the mantle and shrink again to the top cell. For `abe_solid.toml` (1000 km mantle, 100 nodes, mass-coordinate mode) the top cell stays 10.9 km and the largest cell is 14.4 km, the 72nd of 99 cells from the CMB. The cells between the ends are larger than those of the uniform mesh. On an Earth mantle 2891 km thick in radius mode, the largest cell is 1.86 times the uniform cell with 80 nodes (the PROTEUS default) and 2.12 times with 40 nodes (the PROTEUS minimum). In mass-coordinate mode, with the Adams-Williamson density of `abe_solid.toml`, it is 1.56 and 1.77 times the largest cell of the uniform mesh. A 25 km shell with 40 nodes has a uniform cell of 641 m, so the 3000 m default gives the uniform mesh there.

## The CMB cell and the CMB heat flux

In the core module the CMB heat flux is the larger of two values when the core is hotter than the mantle: conduction across the lower half of the first cell, and a boundary-layer law. When the core is not hotter, it is conduction alone ([Core boundary conditions](core_bc.md)). Against uniform meshes of up to 640 nodes, the default mesh with 80 nodes gives a CMB flux within 0.26 % of the finest at 50 and 100 Myr ([the default CMB cell against finer uniform meshes](../Validation/mesh_refinement.md#the-default-cmb-cell-against-finer-uniform-meshes)). This is not an error bound: the uniform flux at 320 nodes still lies 1.8 % above that at 640 nodes at 50 Myr, and 1.4 % at 100 Myr. The case runs with the solid rheology on and the default `mlt_bottom_slope = 0.82`; the slope acts only when the solid rheology is on. A thinner cell costs much more on a molten shell without heat flow through the CMB (see [Limits](#limits)).

## Accepted values

- **0**: that end is not set. With the other end set, its cells grow to keep the node count: on `abe_solid.toml` a 1 km, 200 m or 50 m top cell makes the CMB cell 22.1, 31.0 or 38.5 km in mass-coordinate mode (23.4, 33.0 or 41.1 km in radius mode), against 9.33 km on the uniform mesh. Set the other field too to hold that end.
- **A bool, a string or any other non-number**: refused when the config is read and when the mesh is built, with an error that names the field. A TOML `true` is not read as 1 m.
- **Negative, NaN, infinite or below 2 m**, for either cell: refused when the config is read and when the mesh is built. The 2 m floor is a margin of 2 over the 1 m by which the node solve of the mass-coordinate mode stays inside the boundaries, so that it would clamp a node only below 1 m; at 2 m the CMB half cell also meets the 1 m floor of the `bower2018` core flux.
- **A numeric surface cell of at least the uniform radial cell** $(r_\mathrm{s} - r_\mathrm{c})/(n - 1)$: refused in both coordinate modes. On `abe_solid.toml` it is 10.1 km; the mass-coordinate `"uniform"` top cell is 10.9 km.
- **A CMB cell at or above the CMB cell of the uniform mesh** of the same coordinate mode, or less than 1e-5 of it below: no CMB refinement, and an INFO log line names both values. The CMB cell is then the uniform one with the `"uniform"` or 0 top cell, and that of the surface-refined grid with a set surface cell.
- **Too few nodes**: a set surface cell, or a CMB cell that refines the mesh, needs at least 4 nodes; fewer are refused.
- **`"uniform"`**: accepted only for the surface cell; any other string is refused.

## Limits

- **A thin CMB cell on a molten shell without CMB heat flow**: with a CMB flux of 0 the bottom cells of a molten mantle sit near a zero entropy gradient, where mixing-length convection switches on and off, and a CMB cell much thinner than the uniform cell forces many small solver steps.
- **Below $10^{-7}$ of the mantle thickness**: refused by the grid when the mesh is built. With the 2 m floor this applies only to a mantle thicker than 20,000 km in radius mode, and about the same in mass-coordinate mode, where the fraction of a 2 m cell depends on the density profile.
- **Mass-coordinate mode near the uniform cell**: the uniform end cells in mass coordinates differ from the radial uniform cell. For `abe_solid.toml` the uniform CMB cell is 9.33 km, below the radial 10.1 km, so a CMB cell between the two refines the mesh in radius mode and gives the uniform bottom cell in mass-coordinate mode.
- **Cell density with an external profile** (`eos_method = 2`): on a refined mesh each cell's density is its mass over its volume, in both mesh modes, as on every mass-coordinate mesh. A uniform radius-mode mesh with an external profile takes the mean of the two face densities instead; on the test profile the two differ by up to $10^{-4}$ relative.
- **Each refined mesh**: Aragog checks that the radii increase, that the end radii are the shell radii, and that each set end cell equals its target to $10^{-6}$ relative, or to 2 mm in mass-coordinate mode where that is larger; a violation stops the run with an error.

## Cost

The convergence study on the Validation page gives the number of solver steps up to $10^5$ years on the uniform 100-node mesh and with a 200 m and a 50 m top cell.

**Where this is checked.** [Validation: mesh refinement](../Validation/mesh_refinement.md).

[^cite-vinokur1983]: M. Vinokur, *[On one-dimensional stretching functions for finite-difference calculations](https://doi.org/10.1016/0021-9991(83)90065-7)*, Journal of Computational Physics, 50, 215-234, 1983. [SciX](https://scixplorer.org/abs/1983JCoPh..50..215V/abstract).
