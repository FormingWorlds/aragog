# The core alloy liquidus

The liquidus of the core alloy decides where an inner core freezes: the inner-core boundary lies where the core adiabat meets it. `aragog.core.AlloyLiquidus` gives the liquidus of iron with S, C and O, which lower the melting point, and with Si, H and N, which do not. It has the `t_melt(pressure)` interface of the other melting curves of the core module ([core boundary conditions](core_bc.md)). The `core_module` configuration selects the `iron` or `quadratic` curve; `AlloyLiquidus` is used through the Python API ([tutorial](../Tutorials/core_alloy_liquidus.md)).

## One binary

On the Fe-rich side of an Fe-X binary, pure solid iron is in equilibrium with the liquid alloy. Mori et al. (2017, Eq. 3) and Mashino et al. (2019) describe this liquidus as an ideal solution,

$$
x_\mathrm{Fe} = \exp\left[A \left(1 - \frac{T_\mathrm{Fe}}{T}\right)\right],
$$

with $x_\mathrm{Fe}$ the mole fraction of free Fe in the liquid, $T_\mathrm{Fe}(P)$ the melting curve of pure iron (the PALEOS curve of Anzellini et al. 2013) and $A$ a constant at each pressure. $A$ follows from the eutectic point: at the eutectic composition $w_e$ the liquidus is the eutectic temperature $T_e$, so

$$
A = \frac{\ln x_\mathrm{Fe}(w_e)}{1 - T_\mathrm{Fe}/T_e}.
$$

The liquidus therefore runs from $T_\mathrm{Fe}$ at $w = 0$ to $T_e$ at $w = w_e$. The mole fraction counts Fe against the Fe$_k$X$_m$ units of the alloy, not against single X atoms: Fe$_3$S for S, FeO for O, Fe$_3$C for C below 255 GPa and Fe$_7$C$_3$ above. On the Fe$_3$S units this form reproduces the 3 liquidus curves that Mori et al. (2017, Fig. 9) draw; on S atoms it does not ([validation](../Validation/core_alloy_liquidus.md)). For C the units are not verified: the curves of Mashino et al. (2019, Fig. 8) are closer to straight lines than to this form.

The 2 C units give liquidus temperatures that differ by up to 55 K at 255 GPa. Aragog joins them with a $C^2$ smootherstep over 230 to 280 GPa, so the liquidus has no step; across the blend its slope falls at most 0.5 K/GPa below its value at the blend edges, where the slopes of core liquidus curves are about 4 to 10 K/GPa.

## Several elements

With several elements the mole fraction of free Fe is $x_\mathrm{Fe} = 1 - X$, $X = \sum_i x_i$, and the constant is a harmonic mean of the binary constants weighted by the mole fractions,

$$
\frac{1}{A} = \sum_i \frac{x_i}{X} \frac{1}{A_i}.
$$

For a dilute alloy this gives $T_\mathrm{Fe} - T \approx T_\mathrm{Fe} \sum_i x_i / A_i$, the sum of the dilute depressions of the binaries. Si, H and N lower no melting point: their atoms count with the Fe of the solvent, so an Fe-Si alloy melts like Fe, and Si dilutes the S, C and O units.

A composition past the Fe-rich side of the eutectic, $\sum_i w_i / w_{e,i} > 1$, would crystallise a compound, not iron. There the liquidus is held at its value on the boundary, with the composition scaled to $\sum_i w_i / w_{e,i} = 1$, and `AlloyLiquidus.status` flags the pressure (`past_eutectic`).

## Eutectic data

| Binary | Eutectic temperature | Eutectic composition | Pressures of the data |
|---|---|---|---|
| Fe-S | Simon fit of Mori et al. (2017, p. 138): $T_e = 1348\,\mathrm{K}\,[(P - 21)/36.5 + 1]^{1/2.07}$, $P$ in GPa | curve of Mori et al. (2017, Fig. 5), to 5 wt% S at 330 GPa (Hirose et al. 2021) | 21 to 330 GPa |
| Fe-C | Simon fit of Mashino et al. (2019, p. 138): $T_e = 1513\,\mathrm{K}\,[(P - 5)/8.4 + 1]^{1/4.1}$ | curve of Mashino et al. (2019, Fig. 3), to 3 wt% C at 330 GPa (Mashino et al. 2019, p. 140) | 5 to 330 GPa |
| Fe-O | curve of Oka et al. (2019, Fig. 4) | curve of Oka et al. (2019, Fig. 3), to 325 GPa | 0 to 330 GPa |

The curves were read from the vector paths of the published figures and checked by a second read from a raster image; the tables are in `aragog/core/eutectics/`, each with its source in the header. Outside the pressures of a binary's data the eutectic temperature keeps its ratio to $T_\mathrm{Fe}$ at the end of the data (homologous) and the eutectic composition its end value; `AlloyLiquidus.status` flags such pressures (`extrapolated`) for each depressing element in the alloy.

## Limits

- The ideal form is a fit to the eutectic points, not ideal thermodynamics: for a truly ideal solution $A$ is the entropy of fusion of iron in units of $k_B$, about 1.2, while the S and C binaries give $A$ of 0.1 to 0.9.
- Only the end points of each binary are data; the liquidus between them follows the form. Ternary data (Hirose et al. 2021) are not used to check the harmonic rule.
- H and N have no liquidus data at core pressures and lower no melting point here.
- The solid takes up no light element in this liquidus; solid-liquid partitioning belongs to the core energy budget.
- The slope `AlloyLiquidus.gradient` sets whether a core freezes from the centre: where the liquidus is shallower than the adiabat, iron freezes first at the top ([crystallisation regimes](core_bc.md)).

## References

- Anzellini, S. et al. (2013), Science 340, 464.
- Hirose, K. et al. (2021), Nat. Rev. Earth Environ. 2, 645-658, doi:10.1038/s43017-021-00203-6.
- Mashino, I. et al. (2019), Earth Planet. Sci. Lett. 515, 135-144, doi:10.1016/j.epsl.2019.03.020.
- Mori, Y. et al. (2017), Earth Planet. Sci. Lett. 464, 135-141, doi:10.1016/j.epsl.2017.02.021.
- Oka, K. et al. (2019), Am. Mineral. 104, 1603-1607, doi:10.2138/am-2019-7081.
