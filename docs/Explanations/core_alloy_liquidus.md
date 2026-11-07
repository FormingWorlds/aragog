# The core alloy liquidus

## What this is and why it matters

The liquidus of an alloy is the temperature at which it starts to freeze as it cools. In a planet's core, the inner core grows where the core adiabat meets the liquidus of the core alloy. The adiabat is the temperature profile of a convecting liquid that exchanges no heat with its surroundings. Light elements such as sulfur, carbon and oxygen lower the liquidus of iron, so they move the inner-core boundary. The slope of the liquidus also decides whether the core freezes from the centre or from the top.

## A sketch of the liquidus

![Melting temperature of iron and iron-sulfur alloys against pressure, with an example core adiabat](../figures/alloy_liquidus_sketch_light.svg#only-light){ width="100%" }
![Melting temperature of iron and iron-sulfur alloys against pressure, with an example core adiabat](../figures/alloy_liquidus_sketch_dark.svg#only-dark){ width="100%" }

The figure shows melting temperature against pressure, from the surface to the centre of the Earth at 364 GPa. The solid curves are pure iron and iron with 2 and 4.8 wt% S. The dashed curve is the Fe-S eutectic, the lowest melting point of the mixture. `AlloyLiquidus` computes all 4 curves. The dotted curve is an example core temperature profile, an adiabat from 4500 K at the core-mantle boundary, not a model result. The notes beside the plot explain the 5 numbered points. `tools/verification/figures/sketch_alloy_liquidus.py` builds the figure.

## The model in steps

`aragog.core.AlloyLiquidus` builds the liquidus in three steps. First it treats each binary alone. A binary is iron with one light element X. Then it joins the binaries for an alloy with several light elements.

Last, it takes the eutectic point of each binary from published data. The eutectic point is the temperature $T_e$ and composition $w_e$ at which iron and a compound of X freeze together. It is the lowest point of the liquidus on the iron side.

S, C and O lower the melting point of iron. Si, H and N are treated as if they do not lower it. `AlloyLiquidus` has the `t_melt(pressure)` interface of the other melting curves of the core module ([core boundary conditions](core_bc.md)).

The `core_module` configuration selects the `iron` or `quadratic` curve. `AlloyLiquidus` is used through the Python API ([tutorial](../Tutorials/core_alloy_liquidus.md)).

## One binary

### The ideal-solution form

On the Fe-rich side of an Fe-X binary, pure solid iron is in equilibrium with the liquid alloy. The Fe-rich side is the range of compositions between pure iron and the eutectic. Mori et al. (2017, Eq. 3) and Mashino et al. (2019) describe this liquidus as an ideal solution:

$$
x_\mathrm{Fe} = \exp\left[A \left(1 - \frac{T_\mathrm{Fe}}{T}\right)\right].
$$

Here $x_\mathrm{Fe}$ is the mole fraction of free Fe in the liquid. $T_\mathrm{Fe}(P)$ is the melting curve of pure iron, the PALEOS curve of Anzellini et al. (2013). $A$ is a constant at each pressure.

$A$ follows from the eutectic point. At the eutectic composition $w_e$ the liquidus equals the eutectic temperature $T_e$, so

$$
A = \frac{\ln x_\mathrm{Fe}(w_e)}{1 - T_\mathrm{Fe}/T_e}.
$$

The liquidus therefore runs from $T_\mathrm{Fe}$ at $w = 0$ to $T_e$ at $w = w_e$.

### The units of the alloy

The mole fraction counts Fe against the Fe$_k$X$_m$ units of the alloy, not against single X atoms. The units are:

- S: FeS below 18 GPa and Fe$_3$S above 26 GPa.
- O: FeO.
- C: Fe$_3$C below 255 GPa and Fe$_7$C$_3$ above.

On the Fe$_3$S units this form reproduces the 3 liquidus curves that Mori et al. (2017, Fig. 9) draw. On S atoms it does not ([validation](../Validation/core_alloy_liquidus.md)). For C the units are not verified. The curves of Mashino et al. (2019, Fig. 8) are closer to straight lines than to this form.

### Joining two units

Where the unit changes, Aragog blends the liquidus temperatures of the two units with a $C^2$ smootherstep. A smootherstep is a polynomial weight that rises from 0 to 1 across a pressure window. $C^2$ means that the blend, its slope and its curvature have no step.

**Carbon.** The 2 C units give liquidus temperatures that differ by up to 55 K at 255 GPa. Aragog joins them with a $C^2$ smootherstep over 230 to 280 GPa, so the liquidus has no step.

??? note "Details"
    Across the blend the slope of the liquidus falls at most 0.76 K/GPa (at 2 wt% C) below the lower of its values at 229 and 281 GPa. These two pressures lie just outside the blend. There the slopes are 1.6 to 9.3 K/GPa for 0.5 to 3 wt% C.

**Sulfur.** The sulfide that coexists with iron at the Fe-S eutectic is FeS up to 14 GPa. Above it the sulfide is Fe$_3$S$_2$ (Fei et al. 1997). Fe$_3$S forms between 18 and 21 GPa (Fei et al. 2000).

Fe$_3$S units hold at most 16.06 wt% S. This is less than the S in the eutectic liquid below 17.8 GPa. Aragog therefore uses FeS units up to 18 GPa and Fe$_3$S units from 26 GPa. A $C^2$ smootherstep of the 2 liquidus temperatures joins them over 18 to 26 GPa.

Fe$_3$S$_2$ and Fe$_2$S are not used as units. The two forms run through the same end points but differ between them. The blend therefore adds to the slope of the liquidus near 22 GPa, with no data behind it.

The dilute limit is the limit of a vanishing light-element content, $w \to 0$. Over the window, the depression per wt% S in the dilute limit falls 3.46 times. It falls from 92.72 K at 18 GPa (FeS units) to 26.80 K at 26 GPa (Fe$_3$S units).

??? note "Details"
    Between the end points the two forms differ by up to 357.7 K at 21 GPa (9.15 wt% S). At 18 GPa they differ by 457.7 K (10.32 wt% S), but there the Fe$_3$S form has zero weight.

    Near 22 GPa the blend adds to the slope of the liquidus as follows:

    | S content | Slope added by the blend | Peak slope of the liquidus |
    |---|---|---|
    | 2 wt% | up to 29.6 K/GPa | 51.4 K/GPa |
    | 4 wt% | up to 53.6 K/GPa | 73.9 K/GPa |
    | 8 wt% | up to 81.5 K/GPa | 98.4 K/GPa |
    | 10 wt% | up to 81.8 K/GPa | 96.7 K/GPa |
    | 12 wt% | up to 67.4 K/GPa | 80.0 K/GPa |

### Where the form misses the Fe-S data at low pressure

From 1 bar to 15 GPa the measured Fe-S liquidus is not ideal (Buono & Walker 2011; Chen et al. 2008). The ideal form on FeS units misses the measured points. The RMS misfit is the root mean square of the differences between the form and the points.

Between 10 and 14 GPa the measured liquidus falls faster than the ideal form at 4 and 8 wt% S. The code also puts the onset of top-down freezing at a higher S content than the data. A non-ideal liquidus below 21 GPa, such as the model of Buono & Walker (2011), is not implemented.

??? note "Details"
    The misfit of the ideal form on FeS units:

    | Pressure | Points | RMS misfit | Mean misfit |
    |---|---|---|---|
    | 1 bar | 19 (Friedrich 1910 and Miyazaki 1928) | 78.60 K | $-33.32$ K |
    | 6 GPa | 21 | 153.15 K | $-141.36$ K |
    | 10 GPa | 7 | 88.95 K | $-82.55$ K |
    | 14 GPa | 4 | 150.90 K | $-12.45$ K |

    At 1 bar the form lies above the points up to 11 wt% S and below them from 16 wt% S. These sets hold the points with S on the Fe-rich side of the eutectic.

    The 2 pure-iron points at 1 bar, 1800 and 1809 K, lie 22.4 and 13.4 K below the pure-iron curve at 1 bar (1822.4 K). They test that curve, not the form.

    The change of the liquidus between 10 and 14 GPa. The measured values are linear between the points of Chen et al. (2008).

    | S content | Measured | Ideal form |
    |---|---|---|
    | 4 wt% | $-42.05$ K/GPa | $+3.05$ K/GPa |
    | 8 wt% | $-39.06$ K/GPa | $-14.49$ K/GPa |
    | 12 wt% | $-15.53$ K/GPa | $-26.89$ K/GPa |

    The code liquidus falls from 10 to 14 GPa from 4.61 wt% S. The measured one falls from 1.01 wt% S. This value is linear between the 10 GPa pure-Fe point and the first liquid point at 7.05 wt%. The code therefore puts the onset of top-down freezing at a higher S content than the data.

    The pure iron of the code melts 12.9 and 35.0 K above the pure-Fe points of Chen et al. (2008) at 10 and 14 GPa.

## Several elements

With several elements the mole fraction of free Fe is $x_\mathrm{Fe} = 1 - X$, with $X = \sum_i x_i$. The constant is a harmonic mean of the binary constants, weighted by the mole fractions:

$$
\frac{1}{A} = \sum_i \frac{x_i}{X} \frac{1}{A_i}.
$$

For a dilute alloy this gives $T_\mathrm{Fe} - T \approx T_\mathrm{Fe} \sum_i x_i / A_i$. This is the sum of the dilute depressions of the binaries.

**Silicon.** Si is treated as lowering no melting point. Its atoms count with the Fe of the solvent, so an Fe-Si alloy melts like Fe. Si still dilutes the S, C and O units. For example, 6 wt% Si raises the liquidus of a 2 wt% S alloy by 15 K at 135 GPa.

**Hydrogen and nitrogen.** Aragog has no liquidus data for H and N, so they leave the mole count. An alloy melts like the same alloy without its H and N, with the rest renormalised. It is also held at its eutectic like that alloy.

**Past the eutectic.** A composition past the Fe-rich side of the eutectic, $\sum_i w_i / w_{e,i} > 1$, would crystallise a compound, not iron. There the liquidus is held at its value on the boundary, with the composition scaled to $\sum_i w_i / w_{e,i} = 1$. `AlloyLiquidus.status` flags the pressure (`past_eutectic`).

## Eutectic data

Each binary needs $T_e$ and $w_e$ at every pressure. The data come from different sources at low pressure and at high pressure. Above the data, a rule continues them. The [data sources](#data-sources) table lists every source.

### Low pressure

The data of all 3 binaries start at 1 bar. Below it the eutectic temperature would keep its ratio to $T_\mathrm{Fe}$ at 1 bar. This fixed ratio to the pure-iron melting point is called homologous. The eutectic composition would keep its 1 bar value.

**Fe-S.** Fei et al. (1997, Fig. 2) give the eutectic at 1 bar, 10 and 14 GPa. These points are the nodes. From the nodes, $T_e$ runs towards 1348 K at 21 GPa (Fei et al. 2000), and $w_e$ runs towards the composition curve at 20 GPa.

Through the nodes and these line ends, $T_e$ and $w_e$ follow the monotone cubic Hermite of Fritsch & Carlson (1980). This piecewise cubic stays monotone between the nodes and has no step in slope at them. Over 18 to 21 GPa a smootherstep $b$ of $(P - 18\,\mathrm{GPa})/3\,\mathrm{GPa}$ joins these lines into the high-pressure Simon fit and composition table:

$T_e = (1 - b)\,T_\mathrm{line} + b\,T_\mathrm{fit}$ and the same for $w_e$.

The composition table is continued below 20 GPa along its first segment. $T_e$, $w_e$ and their slopes are continuous at 18 and 21 GPa.

??? note "Details"
    The Fe-S nodes are the means of 2 reads of Fei et al. (1997, Fig. 2), which agree to 1.5 C and 0.03 wt% (`fe_s_low_pressure.csv`):

    | Pressure | $T_e$ | $w_e$ |
    |---|---|---|
    | 1 bar | 1261.25 K | 31.175 wt% S |
    | 10 GPa | 1187.9 K | 20.57 wt% S |
    | 14 GPa | 1135.25 K | 18.305 wt% S |

    The node slopes of the Hermite are:

    - At 1 bar: the first chord.
    - At 10 GPa: the weighted harmonic mean of the 2 secants.
    - At 14 GPa: 0. Here Fe$_3$S$_2$ appears and $T_e$ has its minimum.
    - At the join end: the slope of the fit and of the table.

    The slopes of $T_e$ and $w_e$, and so of the liquidus, have no step at 10 and 14 GPa. A line between the nodes would put a step there with no data behind it (up to 38.6 K/GPa for 12 wt% S at 14 GPa).

    The curve departs from the chord between the nodes. $T_e$ departs by up to $-25.9$ K at 15.91 GPa. $w_e$ departs by up to $-0.52$ wt% at 6.67 GPa. The 6 GPa misfit of the liquidus therefore grows from 142.14 to 153.15 K RMS (22 to 21 points). The eutectic misses Buono & Walker (2011) at 6 GPa by $-42.2$ K and $-1.44$ wt%, instead of $-45.8$ K and $-0.94$ wt%.

    At 18 GPa $w_e$ is 15.50 wt% S. This is below the 16.06 wt% that Fe$_3$S units hold. $w_e$ crosses that value at 17.14 GPa.

**Fe-C.** Below 5 GPa the Fe-C eutectic is a cubic. It runs through 1416.15 K and 4.3 wt% C at 1 bar (Nakajima et al. 2009, p. 207), with zero curvature there. At 5 GPa it takes the value of the Mashino fit and composition table.

??? note "Details"
    At 5 GPa $T_e$ takes the slope of the fit, and $w_e$ the slope 0. The table falls at 5 GPa where the chord from 1 bar rises, so the slope 0 keeps both monotone (Fritsch & Carlson 1980).

    $T_e$ rises by 7.09 K/GPa at 1 bar and 43.93 K/GPa at 5 GPa. It is convex, because the join takes the slope of the Mashino fit at 5 GPa. That slope is 2.1 times the 21.0 K/GPa that Fei & Brosh (2014, p. 158) measure from 5 to 10 GPa.

    $w_e$ runs from 4.300 to 4.311 wt%. Its slope steps from 0 to that of the table at 5 GPa. The liquidus slope of 0.5 to 3 wt% C steps with it, by $-0.37$ to $-2.60$ K/GPa (2 wt% C: from 36.56 to 34.93 K/GPa).

### High pressure

A Simon fit is a power law of pressure for a melting temperature. From 21 GPa the Fe-S eutectic temperature is the Simon fit of Mori et al. (2017, p. 138), $T_e = 1348\,\mathrm{K}\,[(P - 21)/36.5 + 1]^{1/2.07}$, with $P$ in GPa. From 5 GPa the Fe-C eutectic temperature is the Simon fit of Mashino et al. (2019, p. 138), $T_e = 1513\,\mathrm{K}\,[(P - 5)/8.4 + 1]^{1/4.1}$. The Fe-O eutectic temperature is the curve of Oka et al. (2019, Fig. 4).

The eutectic compositions are curves read from published figures. For S and C the read curve ends before 330 GPa. A straight segment then runs from the last table node to an anchor at 330 GPa. The pressures between the end of each read curve and its anchor at 330 GPa count as data.

The eutectic equals the data up to the end of each range.

??? note "Details"
    Mashino et al. (2019, p. 140) give about 3 wt% C for the eutectic at 330 GPa in their text. With that value the eutectic composition falls 4 times faster from 270 to 330 GPa than at the end of their curve. The liquidus of 2 to 3 wt% C then falls with pressure there (down to $-4.6$ K/GPa for 3 wt% C). The end of their 330 GPa liquidus curve in Fig. 8, 3.35 wt%, keeps the slope positive and is used.

### Above the data

Above the end of the data, $P_\mathrm{end}$, the data are taken at an effective pressure $q$. Its slope $dq/dP = 1 - b$ fades from 1 to 0 by a smootherstep $b$ over the 40 GPa above the end of the data. This window is 330 to 370 GPa for S and C, and 325 to 365 GPa for O. The effective pressure is

$q = P_\mathrm{end} + 2h\,[z - z^4 (5/2 - 3z + z^2)]$ with $z = (P - P_\mathrm{end})/2h$ and $h = 20$ GPa, so $q$ ends at $P_\mathrm{end} + h$.

The eutectic temperature there is $T_e(q)\, T_\mathrm{Fe}(P) / T_\mathrm{Fe}(q)$. It keeps its ratio to the pure-iron curve at $q$ (homologous). It does not follow the Simon fit or the Fe-O table at $P$. The composition tables and the Fe-O temperature table continue along their last segment up to $q$.

From 40 GPa above the end, $q$ stays at $P_\mathrm{end} + h$. The eutectic composition keeps its value there: 4.81 wt% S, 3.27 wt% C and 15.28 wt% O. `AlloyLiquidus.status` flags pressures outside the data (`extrapolated`) for each depressing element in the alloy.

The continued compositions move above the data. An alloy whose $\sum_i w_i / w_{e,i}$ passes 1 there therefore crosses its eutectic once, by extrapolation. At the crossing the slope of the liquidus steps up, by 0.3 to 3.7 K/GPa for the alloys tested.

??? note "Details"
    The continued tables move the composition by $-0.19$ wt% S, $-0.08$ wt% C and $+0.22$ wt% O from the end of the data. The continued slope is that of the last table segment. For O this is the last segment of the read curve. For S and C it is the straight segment from the last read node (275 and 270 GPa) to the 330 GPa anchor, not a measured slope.

    The slope of the liquidus changes smoothly into the homologous rule. Between points 0.1 GPa apart over 325 to 380 GPa, it moves by at most 0.025 K/GPa for the alloys tested.

    S and C alloys and their mixtures enter their eutectic above the data, since the S and C tables fall:

    | Alloy | Pressure of the crossing |
    |---|---|
    | 4.85 wt% S | 348 GPa |
    | 4.9 wt% S | 341 GPa |
    | 4.95 wt% S | 335 GPa |
    | 5 wt% S, 3.35 wt% C (the end values) | just above 330 GPa |
    | 3.3 wt% C | 344 GPa |
    | 4 wt% S with 0.65 wt% C | 333 GPa |

    O alloys between 15.07 and 15.28 wt% leave it, since the O table rises: 15.2 wt% at 338 GPa, 15.28 wt% at 354 GPa.

    Inside the data, where the alloy is not past its eutectic, the slope keeps the kinks of the table nodes. The largest kinks, between points 0.1 GPa apart, are of the size of the step where an alloy passes its eutectic:

    - 3.4 K/GPa for 4.8 wt% S at 275 GPa, where the read curve joins the straight segment to the 330 GPa anchor. For the mixture of 2 wt% S, 0.5 wt% C and 3 wt% O the kink there is 1.6 K/GPa.
    - 1.0 K/GPa for 3 wt% C at 25 and 270 GPa.
    - 2.5 K/GPa for 10 wt% O at 55 GPa.

    At the low-pressure Fe-S nodes the slope has no step. The joins at 18 and 21 GPa (S) add none either. The Fe-C join at 5 GPa adds the small step of the composition slope described under [Low pressure](#low-pressure).

    Alloys past their eutectic at low pressure leave it with a step:

    - The alloy of the tutorial is past its eutectic boundary ($\sum_i w_i / w_{e,i} > 1$), with three quarters of the sum from O. It leaves it at 20.54 GPa, where the slope steps from 27.97 to 41.49 K/GPa.
    - 10 wt% O leaves it at 49.75 GPa, where the slope steps from 10.24 to 15.90 K/GPa.

## Data sources

| Source | Binary | Pressure range | What it gives |
|---|---|---|---|
| Fei et al. (1997, Fig. 2) | Fe-S | 1 bar, 10 and 14 GPa | Nodes of $T_e$ and $w_e$. The sulfide at the eutectic: FeS up to 14 GPa, Fe$_3$S$_2$ above. |
| Fei et al. (2000) | Fe-S | 18 to 21 GPa | $T_e$ = 1348 K at 21 GPa, the end of the low-pressure curve. Fe$_3$S forms between 18 and 21 GPa. |
| Mori et al. (2017, p. 138) | Fe-S | from 21 GPa | Simon fit of $T_e$. |
| Mori et al. (2017, Fig. 5) | Fe-S | from 21 GPa, read to 275.8 GPa and tabled to its 275 GPa node | Curve of $w_e$. |
| Hirose et al. (2021) | Fe-S | 330 GPa | Anchor of $w_e$ at 5 wt% S, reached linearly from the 275 GPa node. |
| Nakajima et al. (2009, p. 207) | Fe-C | 1 bar | 1416.15 K and 4.3 wt%, the start of a cubic to 5 GPa. |
| Mashino et al. (2019, p. 138) | Fe-C | from 5 GPa | Simon fit of $T_e$. |
| Mashino et al. (2019, Fig. 3) | Fe-C | from 5 GPa, read to 273 GPa and tabled to its 270 GPa node | Curve of $w_e$. |
| Mashino et al. (2019, Fig. 8) | Fe-C | 330 GPa | Anchor of $w_e$ at 3.35 wt% C, the end of their 330 GPa liquidus, reached linearly from the 270 GPa node. |
| Oka et al. (2019, Fig. 4) | Fe-O | read to 330 GPa | Curve of $T_e$. |
| Oka et al. (2019, Fig. 3) | Fe-O | read to 325 GPa | Curve of $w_e$. |

The data cover 1 bar to 330 GPa for Fe-S and Fe-C, and 0 to 325 GPa for Fe-O.

The curves were read from the vector paths of the published figures. A second read from a raster image checked them. The tables are in `aragog/core/eutectics/`, each with its source in the header.

## Limits

- The ideal form is a fit to the eutectic points, not ideal thermodynamics: its constants give entropies of fusion of 0.1$R$ to 4.4$R$, against 1.16$R$ for iron in the core module.

    ??? note "Details"
        For a truly ideal solution, $A$ is the entropy of fusion of iron over $R$, the gas constant, and is the same for every binary. Over their data the binaries give these values of $A$:

        - S on FeS: 3.50 at 1 bar, 0.45 at 21 GPa.
        - S on Fe$_3$S: 2.47 at 18 GPa, where the blend weight of that form starts from 0. Then 1.79 at 21 GPa, down to 0.21 at 330 GPa.
        - C: from 1.73 (Fe$_3$C, 16 GPa) and 0.50 (Fe$_7$C$_3$, 17 GPa) down to 0.40 and 0.12 at 330 GPa.
        - O: 1.07 (1 bar) to 3.45 (50 GPa).

        The value for iron comes from `ds_fusion` 172.8 J/kg/K of the core module. The dilute slopes of the shipped liquidus, $T_\mathrm{Fe} - T$ per wt% as $w \to 0$:

        | Binary (units) | 1 bar | 21 GPa | 135 GPa | 330 GPa |
        |---|---|---|---|---|
        | Fe-S (FeS, blend at 21 GPa, Fe$_3$S) | 9.1 K | 73.5 K | 124.7 K | 510.4 K |
        | Fe-O (FeO) | 59.3 K | 33.0 K | 66.8 K | 137.7 K |
        | Fe-C (Fe$_3$C, Fe$_7$C$_3$ at 330 GPa) | 54.7 K | 65.7 K | 226.9 K | 792.0 K |

- Close to its eutectic the liquidus falls with pressure where the eutectic composition falls fast, down to $-0.64$ K/GPa. A core in this state freezes from the top, which the core module treats as a regime it does not model ([crystallisation regimes](core_bc.md)).

    ??? note "Details"
        Near the eutectic, $dT/dP \approx dT_e/dP - |\partial T/\partial w|\,|dw_e/dP|$. The liquidus falls with pressure in these cases:

        - Every S content tested from 4.9 to 7 wt%, between 209 and 275 GPa (down to $-0.64$ K/GPa), along the read curve of Mori et al. (2017, Fig. 5).
        - 4 wt% S with 1 wt% C at 270 to 273 GPa ($-0.37$ K/GPa).
        - 3.28 to 3.37 wt% C between 323.8 and 336 GPa (down to $-0.105$ K/GPa for 3.34 wt%), where the Fe-C table and its continuation fall.

- Below 14 GPa the Fe-S eutectic temperature falls with pressure (Fei et al. 1997), and so does the liquidus of S alloys from about 4 wt%, down to $-29.4$ K/GPa.

    ??? note "Details"
        | S content | Pressures where the liquidus falls | Least slope |
        |---|---|---|
        | 4.8 wt% | 10.78 to 13.79 GPa | $-2.4$ K/GPa |
        | 8 wt% | 4.99 to 15.57 GPa | $-16.4$ K/GPa |
        | 12 wt% | 1.74 to 16.41 GPa | $-29.4$ K/GPa |

        Alloys with 1 and 3 wt% S keep rising (least slopes 19.5 and 7.2 K/GPa). The liquidus rises with pressure up to 600 GPa, where the alloy is not past its eutectic, for: up to 4.8 wt% S from 26 GPa, 3.2 wt% C and 10 wt% O from 1 GPa, and the 2 mixtures tested.

- Only the end points of each binary are data, and the liquidus between them follows the form: from 1 bar to 15 GPa the form misses the measured Fe-S liquidus by 79 to 153 K RMS (see [One binary](#one-binary)).
- The blend of the FeS and Fe$_3$S units over 18 to 26 GPa has no data behind it. Near 22 GPa it adds up to 81.8 K/GPa to the slope of the liquidus (see [One binary](#one-binary)).
- Below 5 GPa the Fe-C eutectic has 2 data points, 1 bar and 5 GPa, so the cubic between them is a choice. Lord et al. (2009, p. 158) give 1420 K at 1 bar, 3.85 K above the value used.

    ??? note "Details"
        Nakajima et al. (2009, p. 207) give the 1 bar point as the eutectic between iron and graphite of Bens & Elliott (1961, cited there). Lord et al. (2009, p. 158) place it there too, "possibly metastable". The form keeps Fe$_3$C units there.

- Ternary data (Hirose et al. 2021) are not used to check the harmonic rule.
- The harmonic rule has no ternary eutectic, so a mixture on its eutectic boundary does not melt below the lowest binary eutectic. On a grid of 5151 boundary mixtures of S, C and O the least liquidus is the lowest binary eutectic itself, at the pure binary: 2672 K (S) at 135 GPa and 3713 K (C) at 330 GPa.
- An alloy past its eutectic is held at the boundary by scaling all its S, C and O by one factor. The held liquidus therefore depends on the whole composition, not on a path of the phase diagram. At 135 GPa, adding up to 12 wt% O to 11.2 wt% S (past its 9.35 wt% eutectic) raises it by up to 238 K.
- The `extrapolated` flag starts at the end of the tables, 330 GPa for S and C and 325 GPa for O. This is above the highest pressure of each set of experiments: 254 GPa for Fe-S (Mori et al. 2017, p. 135), 255 GPa for Fe-C (Mashino et al. 2019, p. 135) and 204 GPa for Fe-O (Oka et al. 2019, p. 1603). Above those pressures the eutectic comes from the fits, the read curves and the 330 GPa anchors.
- The blend of the 2 C units has no data behind it. At 255 GPa it lowers the slope of the liquidus by 1.5, 2.0 and 1.3 K/GPa (20 to 42 %) below the mean slope of the 2 unit forms, for 1, 2 and 3 wt% C.
- Aragog has no liquidus data for H and N, and they do not enter the liquidus.
- `AlloyLiquidus` is not a melting curve of the core energy budget. The budget takes `iron` or `quadratic`, and its outer-core enrichment guard reads the light-element fraction of those curves.
- The solid takes up no light element in this liquidus. Solid-liquid partitioning belongs to the core energy budget.
- The slope `AlloyLiquidus.gradient` sets whether a core freezes from the centre. Where the liquidus is shallower than the adiabat, iron freezes first at the top ([crystallisation regimes](core_bc.md)). A liquidus that rises with pressure can still be shallower: at 330 GPa the liquidus of the alloys tested rises by 2.0 to 5.5 K/GPa, against 7.2 K/GPa for the adiabat.

    ??? note "Details"
        The adiabat is that of the default core profile, from the core module defaults: central density 12500 kg/m$^3$, length scale 7272 km, thermal expansivity $1.35 \times 10^{-5}$ K$^{-1}$, heat capacity 840 J/kg/K. The core radius is 3480 km, with 136 GPa at the core-mantle boundary. For 4137 K at the core-mantle boundary the adiabat rises by 6.7 K/GPa at 136 GPa and 7.2 K/GPa at 330 GPa. The liquidus of 2 and 4.8 wt% S, 2 wt% C and the alloy of the tutorial rises by 2.0 to 5.5 K/GPa at 330 GPa.

## References

- Anzellini, S. et al. (2013), Science 340, 464.
- Buono, A. S. and Walker, D. (2011), Geochim. Cosmochim. Acta 75, 2072-2087, doi:10.1016/j.gca.2011.01.030.
- Chudinovskikh, L. and Boehler, R. (2007), Earth Planet. Sci. Lett. 257, 97-103, doi:10.1016/j.epsl.2007.02.024.
- Chen, B., Li, J. and Hauck, S. A. (2008), Geophys. Res. Lett. 35, L07201, doi:10.1029/2008GL033311.
- Fei, Y., Bertka, C. M. and Finger, L. W. (1997), Science 275, 1621-1623, doi:10.1126/science.275.5306.1621.
- Fei, Y. et al. (2000), Am. Mineral. 85, 1830, doi:10.2138/am-2000-11-1229.
- Fei, Y. and Brosh, E. (2014), Earth Planet. Sci. Lett. 408, 155-162, doi:10.1016/j.epsl.2014.09.044.
- Friedrich, K. (1910), Metallurgie 9, 33-48.
- Fritsch, F. N. and Carlson, R. E. (1980), SIAM J. Numer. Anal. 17, 238-246, doi:10.1137/0717021.
- Hansen, M. and Anderko, K. (1958), Constitution of Binary Alloys, 2nd ed., McGraw-Hill, New York, p. 1305.
- Hirose, K. et al. (2021), Nat. Rev. Earth Environ. 2, 645-658, doi:10.1038/s43017-021-00203-6.
- Kamada, S. et al. (2010), Earth Planet. Sci. Lett. 294, 94-100, doi:10.1016/j.epsl.2010.03.011.
- Lord, O. T. et al. (2009), Earth Planet. Sci. Lett. 284, 157, doi:10.1016/j.epsl.2009.04.017.
- Mashino, I. et al. (2019), Earth Planet. Sci. Lett. 515, 135-144, doi:10.1016/j.epsl.2019.03.020.
- Mori, Y. et al. (2017), Earth Planet. Sci. Lett. 464, 135-141, doi:10.1016/j.epsl.2017.02.021.
- Miyazaki, K. (1928), Sci. Rep. Tohoku Imp. Univ. 17, 877-881.
- Morard, G. et al. (2007), Earth Planet. Sci. Lett. 263, 128-139.
- Nakajima, Y. et al. (2009), Phys. Earth Planet. Inter. 174, 202, doi:10.1016/j.pepi.2008.05.014.
- Oka, K. et al. (2019), Am. Mineral. 104, 1603-1607, doi:10.2138/am-2019-7081.
- Ryzhenko, B. and Kennedy, G. C. (1973), Am. J. Sci. 273, 803-810.
- Usselman, T. M. (1975), Am. J. Sci. 275, 278-290.
