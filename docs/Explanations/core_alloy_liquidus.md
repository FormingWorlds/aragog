# The core alloy liquidus

The liquidus of the core alloy decides where an inner core freezes: the inner-core boundary lies where the core adiabat meets it. `aragog.core.AlloyLiquidus` gives the liquidus of iron with S, C and O, which lower the melting point, and with Si, H and N, which are treated as lowering it not at all. It has the `t_melt(pressure)` interface of the other melting curves of the core module ([core boundary conditions](core_bc.md)). The `core_module` configuration selects the `iron` or `quadratic` curve; `AlloyLiquidus` is used through the Python API ([tutorial](../Tutorials/core_alloy_liquidus.md)).

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

The 2 C units give liquidus temperatures that differ by up to 55 K at 255 GPa. Aragog joins them with a $C^2$ smootherstep over 230 to 280 GPa, so the liquidus has no step; across the blend its slope falls at most 0.76 K/GPa (at 2 wt% C) below its value at the blend edges, where the slopes are 1.6 to 9.3 K/GPa for 0.5 to 3 wt% C.

## Several elements

With several elements the mole fraction of free Fe is $x_\mathrm{Fe} = 1 - X$, $X = \sum_i x_i$, and the constant is a harmonic mean of the binary constants weighted by the mole fractions,

$$
\frac{1}{A} = \sum_i \frac{x_i}{X} \frac{1}{A_i}.
$$

For a dilute alloy this gives $T_\mathrm{Fe} - T \approx T_\mathrm{Fe} \sum_i x_i / A_i$, the sum of the dilute depressions of the binaries. Si is treated as lowering no melting point: its atoms count with the Fe of the solvent, so an Fe-Si alloy melts like Fe, and Si dilutes the S, C and O units (6 wt% Si raises the liquidus of a 2 wt% S alloy by 15 K at 135 GPa). Aragog has no liquidus data for H and N, which leave the mole count: an alloy melts, and is held at its eutectic, like the same alloy without its H and N, the rest renormalised.

A composition past the Fe-rich side of the eutectic, $\sum_i w_i / w_{e,i} > 1$, would crystallise a compound, not iron. There the liquidus is held at its value on the boundary, with the composition scaled to $\sum_i w_i / w_{e,i} = 1$, and `AlloyLiquidus.status` flags the pressure (`past_eutectic`).

## Eutectic data

| Binary | Eutectic temperature | Eutectic composition | Pressures of the data |
|---|---|---|---|
| Fe-S | Simon fit of Mori et al. (2017, p. 138): $T_e = 1348\,\mathrm{K}\,[(P - 21)/36.5 + 1]^{1/2.07}$, $P$ in GPa | curve of Mori et al. (2017, Fig. 5), read to 275.8 GPa, then linear to 5 wt% S at 330 GPa (Hirose et al. 2021) | 21 to 330 GPa |
| Fe-C | Simon fit of Mashino et al. (2019, p. 138): $T_e = 1513\,\mathrm{K}\,[(P - 5)/8.4 + 1]^{1/4.1}$ | curve of Mashino et al. (2019, Fig. 3), read to 273 GPa, then linear to 3.35 wt% C at 330 GPa, the end of their 330 GPa liquidus (Fig. 8) | 5 to 330 GPa |
| Fe-O | curve of Oka et al. (2019, Fig. 4), read to 330 GPa | curve of Oka et al. (2019, Fig. 3), read to 325 GPa | 0 to 325 GPa |

The curves were read from the vector paths of the published figures and checked by a second read from a raster image; the tables are in `aragog/core/eutectics/`, each with its source in the header. Mashino et al. (2019, p. 140) give about 3 wt% C for the eutectic at 330 GPa in their text; with that value the eutectic composition falls 4 times faster from 270 to 330 GPa than at the end of their curve, and the liquidus of 2 to 3 wt% C falls with pressure there (down to $-4.6$ K/GPa for 3 wt% C). The end of their 330 GPa liquidus curve in Fig. 8, 3.35 wt%, keeps the slope positive and is used. The pressures between the end of each read curve and its anchor at 330 GPa count as data, and the eutectic equals the data up to the end of each range.

Below the pressures of a binary's data the eutectic temperature keeps its ratio to $T_\mathrm{Fe}$ at the start of the data (homologous) and the eutectic composition its start value. Above them the eutectic composition holds its end value (5 wt% S, 3.35 wt% C, 15.068 wt% O), so an alloy inside its eutectic at the end of the data stays inside above it. The eutectic temperature there is taken at an effective pressure $q$ whose slope $dq/dP = 1 - b$ fades from 1 to 0 by a smootherstep $b$ over the 40 GPa above the end of the data (330 to 370 GPa for S and C, 325 to 365 GPa for O), the Fe-O temperature table continued along its last segment: $q = P_\mathrm{end} + 2h\,[z - z^4 (5/2 - 3z + z^2)]$ with $z = (P - P_\mathrm{end})/2h$ and $h = 20$ GPa, so $q$ ends at $P_\mathrm{end} + h$ and the ratio to $T_\mathrm{Fe}$ is that 20 GPa above the end. `AlloyLiquidus.status` flags pressures outside the data (`extrapolated`) for each depressing element in the alloy. Because the eutectic composition stops changing at the end of the data, the slope of the liquidus steps there. At 330 GPa it rises by 1.3 K/GPa for 1 wt% C, 2.7 K/GPa for 3 wt% C and 3.7 K/GPa for 4.8 wt% S (from 2.0 to 5.7 K/GPa); at 325 GPa it falls by 1.6 K/GPa for 10 wt% O (from 6.2 to 4.6 K/GPa). Elsewhere over 325 to 380 GPa, between points 0.1 GPa apart, it moves by at most 0.012 K/GPa for the alloys tested. Inside the data the slope keeps the kinks of the table nodes, up to 0.13 K/GPa for 10 wt% O.

## Limits

- The ideal form is a fit to the eutectic points, not ideal thermodynamics: for a truly ideal solution $A$ is the entropy of fusion of iron over $R$ and the same for every binary, while the binaries give $A$ of 1.8 (S), 1.7 (Fe$_3$C) and 0.5 (Fe$_7$C$_3$) near 20 GPa, falling to 0.21, 0.40 and 0.12 at 330 GPa, and 1.1 to 3.5 for O.
- Close to its eutectic the liquidus falls with pressure where the eutectic composition falls fast, since there $dT/dP \approx dT_e/dP - |\partial T/\partial w|\,|dw_e/dP|$: for every S content tested from 4.9 to 7 wt%, between 209 and 275 GPa (down to $-0.64$ K/GPa), along the read curve of Mori et al. (2017, Fig. 5), for 4 wt% S with 1 wt% C at 270 to 273 GPa ($-0.37$ K/GPa), and for 3.30 to 3.37 wt% C between 323.8 and 330 GPa (down to $-0.09$ K/GPa for 3.35 wt%), at the end of the Fe-C table. A core in this state freezes from the top, which the core module treats as a regime it does not model ([crystallisation regimes](core_bc.md)). Up to 4.8 wt% S, 3.2 wt% C and 10 wt% O, and for the 2 mixtures tested, the liquidus rises with pressure from 1 to 600 GPa.
- Only the end points of each binary are data; the liquidus between them follows the form. Ternary data (Hirose et al. 2021) are not used to check the harmonic rule.
- Aragog has no liquidus data for H and N, and they do not enter the liquidus.
- `AlloyLiquidus` is not a melting curve of the core energy budget yet: the budget takes `iron` or `quadratic`, and its outer-core enrichment guard reads the light-element fraction of those curves.
- The solid takes up no light element in this liquidus; solid-liquid partitioning belongs to the core energy budget.
- The slope `AlloyLiquidus.gradient` sets whether a core freezes from the centre: where the liquidus is shallower than the adiabat, iron freezes first at the top ([crystallisation regimes](core_bc.md)).

## References

- Anzellini, S. et al. (2013), Science 340, 464.
- Hirose, K. et al. (2021), Nat. Rev. Earth Environ. 2, 645-658, doi:10.1038/s43017-021-00203-6.
- Mashino, I. et al. (2019), Earth Planet. Sci. Lett. 515, 135-144, doi:10.1016/j.epsl.2019.03.020.
- Mori, Y. et al. (2017), Earth Planet. Sci. Lett. 464, 135-141, doi:10.1016/j.epsl.2017.02.021.
- Oka, K. et al. (2019), Am. Mineral. 104, 1603-1607, doi:10.2138/am-2019-7081.
