# Mesh refinement towards the boundaries

How can Aragog use thin cells at the surface and at the core-mantle boundary (CMB) without more nodes?

Aragog can place its nodes closer together near one or both ends of the mantle. You set the thickness of the outermost cell, the innermost cell, or both, in metres; the other cells grow smoothly towards the middle of the mantle. The number of nodes stays the same, so a thin end cell costs no extra memory.

## Why the end cells matter

The heat flow through the surface and through the CMB is computed over the end cells. The CMB flux uses the width of the first cell, and a prescribed surface temperature acts over half the top cell ([Surface boundary conditions](surface_bc.md)). The end cells therefore set the length over which the solver resolves the steep temperature changes at the two boundaries. On a uniform mesh the end cells are as thick as all other cells: 10.1 km for the 1000 km mantle of `abe_solid.toml` with 100 nodes.

## The stretched grid

The mesh starts from a computational coordinate $\xi$, the node index scaled to run uniformly from 0 to 1. A stretching function $s(\xi)$, also from 0 to 1, gives the node positions. The node radii are $r = r_\mathrm{c} + (r_\mathrm{s} - r_\mathrm{c})\, s(\xi)$, with $r_\mathrm{c}$ the CMB radius and $r_\mathrm{s}$ the surface radius, both in m. A uniform mesh has $s = \xi$.

With a target only at the surface, Aragog uses the one-sided hyperbolic tangent

$$
s(\xi) = \frac{\tanh(\beta \xi)}{\tanh \beta},
$$

where $\beta > 0$ is a dimensionless stretching parameter. A larger $\beta$ puts the nodes closer together near $\xi = 1$. Aragog solves for the $\beta$ that makes the last cell, $1 - s(1 - h)$ with $h = 1/(n - 1)$ and $n$ the number of nodes, equal to the target fraction. A target only at the CMB uses the mirror image of the same map.

With targets at both ends, Aragog uses the two-sided function of [Vinokur (1983)](https://doi.org/10.1016/0021-9991(83)90065-7),

$$
s(\xi) = \frac{u}{A + (1 - A)\, u}, \qquad u = \frac{1}{2} \left( 1 + \frac{\tanh\left(\delta (\xi - 1/2)\right)}{\tanh(\delta / 2)} \right),
$$

with two dimensionless parameters: $\delta$ sets how strongly the nodes cluster at both ends, and $A$ sets the balance between the two ends. Aragog solves for $\delta$ and $A$ so that the first and the last cell both equal their targets, and refuses the grid when the solve does not converge or the nodes do not increase.

## Mass-coordinate mode

With `mass_coordinates = true` (the default, and the PROTEUS path), the uniform grid is uniform in the mass coordinate $\xi_m$, not in radius ([Model overview](model.md)). The stretching then acts on $\xi_m$, and Aragog converts each target from metres into a mass-coordinate fraction, so the end cells still have the thickness you set, in radius. The node radii come from the mass-coordinate solve, to 1 mm.

## Worked example

`abe_solid.toml` has a 1000 km mantle and 100 nodes, so the uniform cell is 10.1 km. In radius mode:

- A 1 km top cell makes the bottom cell grow to 23.4 km. Neighbouring cells differ by at most 4.6 %.
- A 1 km top cell and a 2 km CMB cell make the largest cell 21.5 km. Neighbouring cells differ by at most 8.4 %.

In mass-coordinate mode the same settings give a bottom cell of 22.1 km in the first case and a largest cell of 21.5 km in the second.

## Limits

- **0**: that end keeps the spacing of the uniform grid. This is the default for both fields.
- **A bool, a string or any other non-number**: refused when the config is read, with an error that names the field. A TOML `true` is not read as 1 m.
- **Negative, NaN, infinite, or at least the uniform radial cell** $(r_\mathrm{s} - r_\mathrm{c})/(n - 1)$: refused when the config is read.
- **Below $10^{-7}$ of the mantle thickness** (0.1 m for a 1000 km mantle): refused when the mesh is built. Below this fraction the maps no longer hold the end cell.
- **Mass-coordinate mode near the uniform cell**: the uniform end cells in mass coordinates differ from the radial uniform cell. For `abe_solid.toml` the uniform CMB cell is 9.33 km, below the radial 10.1 km, so a CMB cell between the two passes the config check and is refused when the mesh is built, with both field values in the error.
- **Each refined mesh**: Aragog checks that the radii increase, that the end radii are the shell radii, and that each set end cell equals its target to $10^{-6}$ relative (2 mm in mass-coordinate mode); a violation stops the run with an error.

## Cost

The convergence study on the Validation page gives the wall time and the step count for each setting.

**Where this is checked.** [Validation: mesh refinement](../Validation/mesh_refinement.md).
