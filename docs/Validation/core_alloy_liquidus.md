# Core alloy liquidus

## What this page checks

This page checks the liquidus of the core alloy, `aragog.core.AlloyLiquidus`: the temperature at which iron with S, C, O, Si, H and N starts to freeze ([explanation](../Explanations/core_alloy_liquidus.md)). Validation means that a test runs the code on a case where the right answer comes from another source (a curve that a paper draws, a temperature it states, or a second read of a figure) and checks that the code gives that answer within a stated tolerance. A wrong liquidus puts the inner-core boundary of a planet at the wrong radius, and a wrong slope of the liquidus puts the first freezing at the wrong depth. Each case below says why the check exists, what it tests, how we know the right answer and why it matters, and ends with a box "Test details". The tests are in `tests/test_core_alloy_liquidus.py` and, for the pressures below 26 GPa, in `tests/test_core_alloy_liquidus_low_pressure.py`.

## The end points of each binary

**Why this check exists.** Each Fe-X binary (S, C, O) is fixed by 2 points: pure iron and the eutectic. An error in the constant $A$ or in the mole fractions moves the curve off these points, and every alloy liquidus is then wrong.

**What it tests.** At zero light element the liquidus must be the pure-iron melting curve exactly, and at the eutectic composition $w_e$ it must be the eutectic temperature $T_e$ to a relative $10^{-12}$. The test runs at 10, 60, 135, 254, 330 and 450 GPa, also where the eutectic point is extrapolated. A second test checks that Si, H and N alone leave the liquidus at pure iron, that H and N leave an alloy as if they were removed, and that Si dilutes the S units by the mole fractions written out by hand.

**How we know the right answer.** The construction of the ideal form: $A$ is chosen so that the curve passes through the eutectic point.

**Why it matters.** Every alloy liquidus is built from these binaries.

This case has no plot: it checks that the code meets 2 points it is built to meet, an identity with nothing to draw.

??? info "Test details"
    - Reference: the ideal form through pure iron and the eutectic point (analytical).
    - Tolerance: exact at zero light element; relative $10^{-12}$ at the eutectic.
    - Test id: `::test_each_binary_runs_through_pure_iron_and_its_eutectic`, `::test_si_h_and_n_lower_no_melting_point`, `::test_h_and_n_leave_the_mole_basis`, `::test_si_counts_with_the_iron_solvent` (unit tier)
    - Command: `pytest -o "addopts=" tests/test_core_alloy_liquidus.py -k "end or binary_runs or mole_basis or solvent"`

## The Fe-S liquidus curves of Mori et al. (2017)

**Why this check exists.** The ideal form counts Fe against Fe$_3$S units. On S atoms the same form gives other curves, which miss the drawn curves by an RMS of up to 304 K.

**What it tests.** The binary form through the end points of each curve that Mori et al. (2017, Fig. 9) draw for the Fe-rich side of the Fe-Fe$_3$S system, at 30, 130 and 250 GPa. On Fe$_3$S units the root-mean-square (RMS) difference to the curve must be below 5 K (first read) and 6 K (second read); on S atoms it must be above 100 K. A second test runs the shipped `AlloyLiquidus` with its own eutectic data against the same curves.

**How we know the right answer.** The 3 curves of the figure, read twice (see the last case). Mori et al. (2017, p. 138) draw them from their own ideal-solution model through their eutectic points, so this case checks that Aragog reproduces their model curves, not measured liquidus points; their measurements enter through the eutectic points.

**Why it matters.** The S content of a core sets most of its melting point depression.

![Fe-S liquidus against S content for 30, 130 and 250 GPa, Aragog lines and the curves of Mori et al. 2017 as circles](../assets/validation/core_alloy_liquidus_fe_s_mori2017.svg#only-light){ width="70%" }
![Fe-S liquidus against S content for 30, 130 and 250 GPa, Aragog lines and the curves of Mori et al. 2017 as circles](../assets/validation/core_alloy_liquidus_fe_s_mori2017_dark.svg#only-dark){ width="70%" }

**How to read the plot.** The x-axis is the S content in wt%, the y-axis the liquidus in K. The lines are the binary form on Fe$_3$S units through the end points of each curve; the circles are the second read of the curves of Mori et al. (2017). A pass means that the circles lie on the lines at all 3 pressures; at 30 GPa the curve bends down towards the eutectic, which the Fe$_3$S units give and S atoms do not.

On Fe$_3$S units the form misses the curves by an RMS of 2.06, 2.22 and 0.70 K (first read) and 3.61, 3.27 and 1.37 K (second read) at 30, 130 and 250 GPa; on S atoms it misses by 304, 193 and 136 K. The shipped `AlloyLiquidus` misses the first read by 2.6, 2.0 and 4.6 K.

??? info "Test details"
    - Reference: Mori et al. (2017), Fig. 9, 2 reads (`tests/validation/data/core_eutectics/mori_liquidus_*GPa.csv`).
    - Tolerance: RMS below 5 K (first read) and 6 K (second read); above 100 K on S atoms; the shipped liquidus below the stated RMS plus 1 K.
    - Test id: `::test_the_fe_s_liquidus_curves_of_mori_are_the_ideal_form_on_fe3s_units`, `::test_the_shipped_liquidus_misses_the_figure_curves_as_stated` (unit tier)
    - Command: `pytest -o "addopts=" tests/test_core_alloy_liquidus.py -k "mori or shipped_liquidus"`

## The Fe-C liquidus curves of Mashino et al. (2019)

**Why this check exists.** The C units (Fe$_3$C below 255 GPa, Fe$_7$C$_3$ above) are not verified. This case states how far the ideal form is from the curves the paper draws and keeps the misfit from growing.

**What it tests.** The binary form through the end points of each curve of Mashino et al. (2019, Fig. 8) at 54, 138, 203, 255 and 330 GPa; the zero-C end is taken at $w = 0$ by extrapolation of the first 2 points of the curve. The RMS difference must stay below the stated value plus 1 K. A second test runs the shipped `AlloyLiquidus` with its own eutectic data against the same curves.

**How we know the right answer.** The 5 curves of the figure, read twice (see the last case).

**Why it matters.** C lowers the melting point of a core less than S, but its liquidus between pure iron and the eutectic is the least certain part of the model.

![Fe-C liquidus against C content for 5 pressures, Aragog lines and the curves of Mashino et al. 2019 as circles](../assets/validation/core_alloy_liquidus_fe_c_mashino2019.svg#only-light){ width="70%" }
![Fe-C liquidus against C content for 5 pressures, Aragog lines and the curves of Mashino et al. 2019 as circles](../assets/validation/core_alloy_liquidus_fe_c_mashino2019_dark.svg#only-dark){ width="70%" }

**How to read the plot.** The x-axis is the C content in wt%, the y-axis the liquidus in K. The lines are the binary form through the end points of each curve; the circles are points of the curves of Mashino et al. (2019, first read). Between the end points the circles lie below the lines: the drawn curves are close to straight lines, and the ideal form lies above them.

The form misses the curves by an RMS of 48.7, 110.1, 120.8, 114.5 and 101.5 K at 54, 138, 203, 255 and 330 GPa, and lies above them by at most 69, 154, 173, 159 and 142 K; straight lines between the end points miss by 0.0, 38.9, 42.6, 84.8 and 112.4 K. The shipped `AlloyLiquidus` misses the curves by 46.5, 175.1, 103.5, 105.1 and 70.1 K: its eutectic points come from the Simon fit and the composition curve of the same paper and differ from the ends of the drawn curves (at 138 GPa the fit gives 3012 K, the curve ends at 2885 K).

**Known limit.** Between pure iron and the eutectic the liquidus of a C-bearing alloy differs from the drawn curves by an RMS of 47 to 175 K.

??? info "Test details"
    - Reference: Mashino et al. (2019), Fig. 8, 2 reads (`tests/validation/data/core_eutectics/mashino_liquidus_*GPa.csv`).
    - Tolerance: RMS below the stated value plus 1 K (form and shipped liquidus).
    - Test id: `::test_the_fe_c_misfit_against_mashino_stays_as_stated`, `::test_the_shipped_liquidus_misses_the_figure_curves_as_stated` (unit tier)
    - Command: `pytest -o "addopts=" tests/test_core_alloy_liquidus.py -k "mashino or shipped_liquidus"`

## Eutectic temperatures

**Why this check exists.** The eutectic temperature fixes the lowest point of each binary. A wrong fit constant or a wrong table shifts the whole liquidus of an alloy near its eutectic.

**What it tests.** The Fe-S eutectic temperature at 254 GPa and the Fe-C eutectic temperature at 330 GPa against the values the papers state, within 50 K, and the Fe-C value against the Simon fit of Mashino et al. (2019), retyped in the test, to a relative $10^{-3}$; and the Fe-O eutectic temperature from 0 to 325 GPa, the end of the Fe-O data, against the second read of the curve of Oka et al. (2019, Fig. 4), the read that the table does not come from, within the drawn line width (0.354 pt) in figure points.

**How we know the right answer.** Mori et al. (2017) measured 3550 K at 254 GPa (run 11); Mashino et al. (2019, p. 140) give about 3750 K at 330 GPa; Oka et al. (2019) draw the Fe-O curve.

**Why it matters.** Near the eutectic the liquidus of an alloy follows the eutectic temperature.

![Eutectic temperatures of Fe-S, Fe-C and Fe-O against pressure with the values of Mori et al. 2017, Mashino et al. 2019 and Oka et al. 2019](../assets/validation/core_alloy_liquidus_eutectic_temperatures.svg#only-light){ width="70%" }
![Eutectic temperatures of Fe-S, Fe-C and Fe-O against pressure with the values of Mori et al. 2017, Mashino et al. 2019 and Oka et al. 2019](../assets/validation/core_alloy_liquidus_eutectic_temperatures_dark.svg#only-dark){ width="70%" }

**How to read the plot.** The x-axis is the pressure in GPa, the y-axis the eutectic temperature $T_e$ in K. The lines are the eutectic temperatures of the 3 binaries in Aragog; the circles are points of the second read of Oka et al. (2019), the square the value of Mori et al. (2017) and the diamond the value of Mashino et al. (2019). A pass means that the square and the diamond lie at most 50 K from their lines and the circles on the Fe-O line. The code gives 3541.1 K (Fe-S, 254 GPa) and 3713.3 K (Fe-C, 330 GPa). Up to the end of each range the eutectic equals the shipped tables and, at 330 GPa, the anchors (5 wt% S, 3.35 wt% C and the Fe-C fit).

??? info "Test details"
    - Reference: Mori et al. (2017), run 11; Mashino et al. (2019), p. 140; Oka et al. (2019), Fig. 4, read B.
    - Tolerance: 50 K (Fe-S, Fe-C); relative $10^{-3}$ against the Fe-C fit; 0.354 pt perpendicular distance in figure points (Fe-O).
    - Test id: `::test_the_eutectic_temperatures_meet_the_papers`, `::test_the_fe_o_eutectic_temperature_meets_the_second_read` (unit tier)
    - Command: `pytest -o "addopts=" tests/test_core_alloy_liquidus.py -k "eutectic_temperature"`

## Several light elements

**Why this check exists.** With several elements the constants $A_i$ of the binaries are combined. A mean of the wrong kind gives a depression that is too small by up to half at 330 GPa.

**What it tests.** An alloy with 2 wt% S, 0.5 wt% C and 3 wt% O at 135 and 330 GPa. The depression $T_\mathrm{Fe} - T$ must equal the harmonic rule computed by hand to a relative $10^{-12}$ and the stated values within 0.1 K, and lie between the composition-weighted rule and the sum of the binary depressions. Scaled down by 10, 100 and 1000, the depression over the dilute sum $T_\mathrm{Fe} \sum_i x_i / A_i$ must tend to 1.

**How we know the right answer.** The rules written out in the test and the dilute limit of the ideal form (analytical).

**Why it matters.** A core holds several light elements at once.

![Melting point depression against total light-element content at 135 GPa for the 4 rules](../assets/validation/core_alloy_liquidus_rules_135gpa.svg#only-light){ width="70%" }
![Melting point depression against total light-element content at 135 GPa for the 4 rules](../assets/validation/core_alloy_liquidus_rules_135gpa_dark.svg#only-dark){ width="70%" }

![Melting point depression against total light-element content at 330 GPa for the 4 rules](../assets/validation/core_alloy_liquidus_rules_330gpa.svg#only-light){ width="70%" }
![Melting point depression against total light-element content at 330 GPa for the 4 rules](../assets/validation/core_alloy_liquidus_rules_330gpa_dark.svg#only-dark){ width="70%" }

**How to read the plot.** The x-axis is the total S + C + O content in wt% at a fixed ratio of 4 : 1 : 6 by mass, the y-axis the depression in K, at 135 GPa (top) and 330 GPa (bottom). The solid line is the harmonic rule of Aragog, the dashed lines 3 other rules. All rules agree at small content; the harmonic rule stays close to the sum of the binaries and far from the composition-weighted rule.

The depression is 640.7 K at 135 GPa and 1748.6 K at 330 GPa. The composition-weighted rule gives 488.7 and 886.6 K, the sum of the binaries 664.2 and 1983.4 K, the dilute sum 679.6 and 2207.9 K. Scaled down by 10, 100 and 1000, the depression over the dilute sum is 0.99462, 0.99947 and 0.99995.

??? info "Test details"
    - Reference: the 4 rules and the dilute limit, written out in the test (analytical).
    - Tolerance: relative $10^{-12}$ against the harmonic rule; 0.1 K against the stated values; ratio at most $10^{-3}$ from 1 at the smallest content.
    - Test id: `::test_the_harmonic_rule_gives_the_stated_depressions`, `::test_the_harmonic_rule_meets_the_dilute_limit` (unit tier)
    - Command: `pytest -o "addopts=" tests/test_core_alloy_liquidus.py -k harmonic`

## The slope of the liquidus

**Why this check exists.** The slope $dT/dP$ of the liquidus decides where a core starts to freeze: where it is smaller than the slope of the core adiabat, iron freezes first at the top. A kink in a table, an anchor point or the end of the data can make a false slope there.

**What it tests.** For C contents of 0.5 to 3.2 wt%, S up to 4.8 wt%, O up to 10 wt% and 2 mixed alloys, the slope must be positive at every pressure up to 600 GPa where the alloy is not past its eutectic, from 26 GPa for alloys with S and from 1 GPa for the others; below 26 GPa the case "The low-pressure eutectic" pins the slope of S alloys. Over 325 to 380 GPa, into the extrapolation above the data, the slope must change by less than 0.1 K/GPa between points 0.1 GPa apart. Alloys whose sum of $w_i / w_{e,i}$ passes 1 above the data must cross their eutectic once, at the pinned pressure and with the pinned slope step: S and C alloys and an S-C mixture into it, O alloys out of it. Across the blend of the 2 C units the slope must fall less than 1 K/GPa below the lower of its values at 229 and 281 GPa, just outside the blend. For 4.9 to 7 wt% S and for 4 wt% S with 1 wt% C the least slope and the pressures where the slope is negative are pinned, and outside them the slope must be positive.

**How we know the right answer.** A liquidus of a fixed composition rises with pressure where its data do. The pinned values are the code's own output on the read curve of Mori et al. (2017, Fig. 5): they show a later change, not an independent answer, and these tests carry `physics_invariant`, not `reference_pinned`.

**Why it matters.** A false negative slope gives a top-down crystallisation regime in a coupled run.

![Slope of the liquidus against pressure from 150 to 400 GPa for 4 alloys](../assets/validation/core_alloy_liquidus_slope.svg#only-light){ width="70%" }
![Slope of the liquidus against pressure from 150 to 400 GPa for 4 alloys](../assets/validation/core_alloy_liquidus_slope_dark.svg#only-dark){ width="70%" }

**How to read the plot.** The x-axis is the pressure in GPa, the y-axis the slope $dT/dP$ in K/GPa, for 2 wt% C, 4.8 and 5.4 wt% S and the mixed alloy of the tutorial (S 2, C 0.5, O 3, Si 4 wt%). The dashed line is 0. A pass means that every line except 5.4 wt% S stays above 0; 5.4 wt% S dips below 0 at 259 to 275 GPa. The small kinks are the 5 GPa nodes of the composition tables; the larger kinks at 275 GPa (S) and 270 GPa (C) are the nodes where the read curves join the straight segments to the 330 GPa anchors; above the data the slopes join the homologous rule without a step.

The least slope of 3.2 wt% C is +0.12 K/GPa, and over 325 to 380 GPa the slope changes by at most 0.025 K/GPa (3 wt% C) between points 0.1 GPa apart. Across the C blend the slope falls at most 0.76 K/GPa (2 wt% C) below the lower of its values at 229 and 281 GPa; a 10 GPa blend would fall 3.7 to 8.9 K/GPa.

**Known limit.** Close to its eutectic the liquidus falls with pressure where the eutectic composition falls fast: 4.9 wt% S at 274.4 to 274.9 GPa (least slope $-0.036$ K/GPa), 5.0 wt% at 272 to 275 GPa ($-0.16$ K/GPa), 5.4 wt% at 259 to 275 GPa ($-0.64$ K/GPa), 5.8 wt% at 244 to 261 GPa ($-0.51$ K/GPa), 6 wt% at 238.5 to 252.1 GPa ($-0.44$ K/GPa), 7 wt% at 208.9 to 209.9 GPa ($-0.072$ K/GPa), 4 wt% S with 1 wt% C at 270 to 273 GPa ($-0.37$ K/GPa), and 3.28 to 3.37 wt% C between 323.8 and 336 GPa (down to $-0.105$ K/GPa for 3.34 wt%). Above the data, alloys cross their eutectic by extrapolation of the composition tables: into it 4.85, 4.9 and 4.95 wt% S at 348, 341 and 335 GPa, 3.3 wt% C at 344 GPa and 4 wt% S with 0.65 wt% C at 333 GPa; out of it 15.2 and 15.28 wt% O at 338 and 354 GPa.

??? info "Test details"
    - Reference: the sign of the slope (physical); for 4.9 to 7 wt% S, pinned values of the code's own output on the read curve of Mori et al. (2017, Fig. 5).
    - Tolerance: slope above 0; step below 0.1 K/GPa; crossing pressures to 0.02 GPa; blend dip below 1 K/GPa; pinned least slope to 0.01 K/GPa and the negative range within 0.2 GPa.
    - Test id: `::test_the_liquidus_rises_with_pressure`, `::test_the_end_of_the_data_adds_no_slope_step`, `::test_alloys_near_the_end_eutectic_cross_it_above_the_data`, `::test_the_fe_c_liquidus_falls_near_the_end_of_its_data_for_3_28_to_3_37_wt_percent`, `::test_the_carbon_unit_switch_adds_no_slope_dip`, `::test_near_the_eutectic_the_liquidus_falls_where_the_read_curve_falls`, `::test_the_liquidus_slope_against_pressure` (unit tier)
    - Command: `pytest -o "addopts=" tests/test_core_alloy_liquidus.py -k "slope or rises or falls or cross"`; the plot: `pytest -o "addopts=" tests/test_core_alloy_liquidus.py -k slope_against_pressure --validation-plots`

## The Fe-S liquidus from 1 bar to 15 GPa

**Why this check exists.** Below 21 GPa the liquidus uses the ideal form on FeS units through measured eutectic points. The measured liquidus at these pressures is not ideal: it has a negative curvature and, at 14 GPa, 2 inflection points (Buono & Walker 2011; Chen et al. 2008). This case states how far the form is from measured liquidus points and keeps the misfit from growing.

**What it tests.** The shipped `AlloyLiquidus` for Fe-S against 21 measured liquidus points at 1 bar (Friedrich 1910 and Miyazaki 1928, as compiled by Hansen & Anderko 1958 and drawn by Buono & Walker 2011, Fig. 6; the 2 points past the 1 bar eutectic left out), 23 at 6 GPa (Buono & Walker 2011, Table 1, without the eutectic run) and against 7 and 4 liquidus points at 10 and 14 GPa (Chen et al. 2008, Fig. 1, read A). The RMS misfit must stay below the stated value plus 1 K and the mean misfit must differ from the stated value by less than 1 K. A second test pins the change of the liquidus from 10 to 14 GPa at 4, 8 and 12 wt% S against the same points.

**How we know the right answer.** Measured liquidus temperatures: the 1 bar points read twice (vector marker centres and a 600 dpi raster, which agree to 0.30 pt of the figure, 1.3 K and 0.0002 in mole fraction), run temperatures with $\pm 25$ K and microprobe compositions of the liquid at 6 GPa, and the points of Chen et al. (2008, Fig. 1) read twice (vector symbols and a 600 dpi raster, which agree to 0.27 wt% and 3.1 K). None of these points sets a node of the eutectic.

**Why it matters.** The liquidus below 21 GPa decides when the core of a planet smaller than the Earth (Mars, about 20 to 40 GPa in the core) starts to freeze.

![Fe-S liquidus against S content at 6, 10 and 14 GPa, Aragog lines and measured points as circles](../assets/validation/core_alloy_liquidus_low_pressure_liquidus.svg#only-light){ width="70%" }
![Fe-S liquidus against S content at 6, 10 and 14 GPa, Aragog lines and measured points as circles](../assets/validation/core_alloy_liquidus_low_pressure_liquidus_dark.svg#only-dark){ width="70%" }

**How to read the plot.** The x-axis is the S content in wt%, the y-axis the liquidus in K. Each line is the shipped liquidus at one pressure, from pure iron to the eutectic; the circles in the same colour are the measured points. The lines lie below most points: the ideal form does not follow the measured curvature.

![Fe-S liquidus at 1 bar against S content, Aragog line and the points of Friedrich 1910 and Miyazaki 1928](../assets/validation/core_alloy_liquidus_one_bar.svg#only-light){ width="70%" }
![Fe-S liquidus at 1 bar against S content, Aragog line and the points of Friedrich 1910 and Miyazaki 1928](../assets/validation/core_alloy_liquidus_one_bar_dark.svg#only-dark){ width="70%" }

The second plot shows 1 bar: the line is the shipped liquidus from pure iron to the 1 bar eutectic, the triangles and squares the 2 data sets. The line lies above the points up to 11 wt% S (by up to 67 K) and below them from 16 wt% S to the eutectic (by up to 135 K).

The RMS misfit is 75.0 K at 1 bar (mean $-28.4$ K, largest 135.0 K), 141.4 K at 6 GPa (mean $-130.6$ K, largest 227.8 K), 89.0 K at 10 GPa (mean $-82.6$ K) and 150.9 K at 14 GPa (mean $-12.4$ K). From 10 to 14 GPa the measured liquidus falls by 42.05, 39.06 and 15.53 K/GPa at 4, 8 and 12 wt% S; the shipped liquidus changes by $+3.0$, $-14.5$ and $-26.9$ K/GPa.

**Known limit.** A non-ideal liquidus below 21 GPa, such as the model of Buono & Walker (2011), is not implemented; the misfit above is the size of that limit.

??? info "Test details"
    - Reference: Buono & Walker (2011), Fig. 6, p. 2079, 2 reads, and Table 1, p. 2075; Chen et al. (2008), Fig. 1, 2 reads (`tests/validation/data/core_eutectics/low_pressure/`).
    - Tolerance: RMS below the stated value plus 1 K, mean to 1 K; the 2 reads of the 1 bar points to 0.5 pt; slopes to 0.05 K/GPa (code) and 0.005 K/GPa (data).
    - Test id: `test_core_alloy_liquidus_low_pressure.py::test_the_low_pressure_liquidus_misses_the_measured_points_as_stated`, `::test_the_two_reads_of_the_1_bar_points_agree`, `::test_from_10_to_14_gpa_the_liquidus_falls_less_than_the_data` (unit tier)
    - Command: `pytest -o "addopts=" tests/test_core_alloy_liquidus_low_pressure.py -k "misses or 1_bar or falls_less"`; the plots: `pytest -o "addopts=" tests/test_core_alloy_liquidus_low_pressure.py -k "1_bar or falls_less" --validation-plots`

## The low-pressure eutectic

**Why this check exists.** Below the Simon fits (21 GPa for S, 5 GPa for C) the eutectic comes from a few measured points joined into the fits. A wrong node or a step at a join changes the liquidus of every alloy there and its slope.

**What it tests.** The Fe-S eutectic at 1 bar, 10 and 14 GPa must be the mean of the 2 reads of Fei et al. (1997, Fig. 2), to a relative $10^{-12}$, and the 2 reads must agree to 1.6 C and 0.035 wt%. $T_e$ and $w_e$ must be linear between the nodes, with the stated slopes on each side of 10 and 14 GPa, and continuous in value and slope at 18 and 21 GPa (S) and 5 GPa (C); inside 18 to 21 GPa the eutectic must be the smootherstep mean of the line and the fit. The Fe-C eutectic below 5 GPa must be the cubic written out by hand. The eutectic must miss each check point by the stated amount: 10 temperatures and 9 compositions from 1 bar to 25 GPa. At the 2 anchors of the fits (1348 K at 21 GPa, 1513 K at 5 GPa) only the composition is a check.

**How we know the right answer.** The nodes are 2 reads of a measured phase diagram. The check points are measured eutectics: at 6 GPa Buono & Walker (2011), Ryzhenko & Kennedy (1973), Usselman (1975) and Morard et al. (2007), all from Table 1 of Buono & Walker (2011); at 21 GPa Fei et al. (2000, p. 1831); at 21 and 25 GPa the linear fit of Kamada et al. (2010, p. 98); at 21.9 and 23.0 GPa quench runs with a liquid (Chudinovskikh & Boehler 2007, Table 1), upper bounds of the eutectic temperature, and at 21.9 GPa a liquid on the Fe-rich liquidus, whose S content bounds the eutectic composition from below; for Fe-C, 1 bar (Lord et al. 2009, p. 158) and 5 and 10 GPa (Fei & Brosh 2014, p. 158).

**Why it matters.** The eutectic is the end point of every Fe-S and Fe-C liquidus at these pressures, and its slope sets the slope of the liquidus near it.

![Eutectic temperature of Fe-S and Fe-C against pressure from 1 bar to 40 GPa, with the nodes and the check points](../assets/validation/core_alloy_liquidus_low_pressure_eutectic_t.svg#only-light){ width="65%" }
![Eutectic temperature of Fe-S and Fe-C against pressure from 1 bar to 40 GPa, with the nodes and the check points](../assets/validation/core_alloy_liquidus_low_pressure_eutectic_t_dark.svg#only-dark){ width="65%" }

![Eutectic liquid composition of Fe-S and Fe-C against pressure from 1 bar to 40 GPa, with the nodes and the check points](../assets/validation/core_alloy_liquidus_low_pressure_eutectic_w.svg#only-light){ width="65%" }
![Eutectic liquid composition of Fe-S and Fe-C against pressure from 1 bar to 40 GPa, with the nodes and the check points](../assets/validation/core_alloy_liquidus_low_pressure_eutectic_w_dark.svg#only-dark){ width="65%" }

**How to read the plots.** The x-axis is the pressure in GPa, the y-axis the eutectic temperature in K (first plot) or the S or C content of the eutectic liquid in wt% (second plot). The lines are the shipped eutectic of Fe-S and Fe-C, the filled circles the Fe-S nodes, the open squares the check points in the colour of their binary, with their stated temperature errors as bars. The Fe-S line falls to its minimum at 14 GPa and rises into the Simon fit over 18 to 21 GPa.

Code minus check point, in temperature: $-45.8$ K (Buono & Walker and Ryzhenko & Kennedy, 1263 K), $-41.8$ K (Usselman), $+77.2$ K (Morard et al., $\pm 170$ K) at 6 GPa; $+36.6$ and $+52.4$ K against the fit of Kamada et al. at 21 and 25 GPa, inside its errors of 54.7 and 57.5 K; 148.0 and 66.8 K below the quench runs at 21.9 and 23.0 GPa; for Fe-C, $-3.85$ K at 1 bar and $+77.5$ K at 10 GPa (the Mashino fit). In composition: $-0.94$ wt% S against Buono & Walker at 6 GPa, $-0.67$ wt% against Fei et al. (2000) at 21 GPa, $+1.68$ wt% above the 13.0 $\pm$ 1.1 wt% S of the liquid at 21.9 GPa (a lower bound, as it should be), and $+0.31$ wt% C against Fei & Brosh at 5 GPa.

**Known limit.** The 6 GPa eutectic of the nodes lies 46 K and 0.9 wt% S below the 6 GPa measurements: the nodes at 1 bar and 10 GPa come from one phase diagram, and the line between them does not hold the eutectic temperature constant to 6 GPa as Buono & Walker (2011) find. Below 14 GPa the liquidus of S alloys from about 4 wt% falls with pressure, as the eutectic does: 4.8 wt% S at 9.33 to 13.99 GPa (least slope $-5.8$ K/GPa), 8 wt% at 5.49 to 14.00 GPa ($-20.0$ K/GPa) and 12 wt% at 2.08 to 14.00 GPa ($-34.3$ K/GPa). The blend of FeS and Fe$_3$S units over 18 to 26 GPa adds up to 29.6, 53.6 and 81.5 K/GPa to the slope near 22 GPa for 2, 4 and 8 wt% S, with no data behind it.

??? info "Test details"
    - Reference: Fei et al. (1997), Fig. 2, 2 reads; the check points in `tests/validation/data/core_eutectics/low_pressure/low_pressure_eutectic_checks.csv`, each with its source and page; the cubic and the smootherstep written out by hand; the unit blend and the falling ranges are the code's own output (`physics_invariant`).
    - Tolerance: nodes and cubic to a relative $10^{-12}$; slopes to $2 \times 10^{-3}$ K/GPa; misses to 0.02 K and 0.002 wt%; Kamada et al. inside their errors; quench runs above the eutectic; the 21.9 GPa liquid below $w_e$; anchors: composition only; blend slope to 0.2 K/GPa; falling ranges to 0.015 GPa.
    - Test id: `test_core_alloy_liquidus_low_pressure.py::test_the_fe_s_eutectic_nodes_are_the_mean_of_two_reads_of_fei_1997`, `::test_the_fe_s_eutectic_is_linear_between_the_nodes_with_kinks_at_10_and_14_gpa`, `::test_the_eutectic_joins_the_fits_with_a_continuous_slope`, `::test_the_fe_s_eutectic_inside_the_join_is_the_smootherstep_mean`, `::test_the_fe_c_eutectic_below_5_gpa_is_the_natural_cubic`, `::test_the_low_pressure_eutectic_against_the_check_points`, `::test_the_unit_blend_adds_the_stated_slope`, `::test_below_26_gpa_the_s_liquidus_falls_where_the_eutectic_falls` (unit tier)
    - Command: `pytest -o "addopts=" tests/test_core_alloy_liquidus_low_pressure.py`; the plots: `pytest -o "addopts=" tests/test_core_alloy_liquidus_low_pressure.py -k check_points --validation-plots`

## The 2 reads of each figure curve

**Why this check exists.** Every curve taken from a figure (the eutectic compositions of Fe-S, Fe-C and Fe-O, the Fe-O eutectic temperature and the 8 liquidus curves) is a read of a drawing. A read can follow the wrong line or drift off it.

**What it tests.** Each curve is read twice: from the vector path of the curve in the PDF (read A) and from a 300 dpi raster image (read B). The largest perpendicular distance from a point of read B to read A must be within the tolerance of the file, in points of the source figure. A second test checks that each eutectic table shipped with Aragog lies on read A of its figure.

**How we know the right answer.** 2 independent reads of the same drawing.

**Why it matters.** The eutectic tables of Aragog and the reference curves of the cases above come from these reads.

![Largest distance between the 2 reads of each figure curve over its tolerance](../assets/validation/core_alloy_liquidus_two_reads.svg#only-light){ width="60%" }
![Largest distance between the 2 reads of each figure curve over its tolerance](../assets/validation/core_alloy_liquidus_two_reads_dark.svg#only-dark){ width="60%" }

**How to read the plot.** Each bar is one data file: X and a pressure in GPa for a liquidus curve, "X w" for a eutectic composition and "O T" for the Fe-O eutectic temperature. The bar length is the largest distance between the 2 reads over the tolerance of the file, half the drawn line width (the full width for the thin Fe-O temperature line). A pass means that every bar ends left of the dashed line at 1 (the test allows 0.02 pt above the tolerance). The largest distance is 0.07 to 1.35 pt against tolerances of 0.35 to 2.34 pt, at most 0.80 of the tolerance.

**Known limit.** The second read of the Fe-O composition has no points from 42 to 52 GPa, where an open symbol in the colour of the curve lies on it.

??? info "Test details"
    - Reference: read A and read B of each curve (`tests/validation/data/core_eutectics/`).
    - Tolerance: the tolerance in the header of each file, plus 0.02 pt; shipped tables equal to read A within $6 \times 10^{-6}$ (mass fraction) or 0.06 K.
    - Test id: `::test_the_two_reads_of_each_figure_curve_agree`, `::test_the_two_reads_figure`, `::test_the_shipped_tables_are_the_first_read` (unit tier)
    - Command: `pytest -o "addopts=" tests/test_core_alloy_liquidus.py -k "reads or first_read"`

## Related tests

The other tests in `tests/test_core_alloy_liquidus.py` check that a composition past the eutectic is held at its boundary and flagged, that at 255 GPa the liquidus is the mean of its Fe$_3$C and Fe$_7$C$_3$ forms, that the gradient of an alloy without S, C and O passes `jax.debug_nans`, that above the data the eutectic composition continues its last table segment while the eutectic temperature becomes homologous, both flagged, that O is flagged beyond its composition table at 325 GPa, that the composition slope at zero light element is the dilute slope, that a traced or 0-d array composition works like a float, that invalid compositions and a binary with $A \le 0$ are refused, and, with Hypothesis, that every valid alloy melts at or below pure iron and that more S never raises the liquidus on the Fe-rich side. The other tests in `tests/test_core_alloy_liquidus_low_pressure.py` check that the S and C data start at 1 bar, so nothing below 21 or 5 GPa is flagged, that below 1 bar the eutectic is held homologous and flagged, that the Fe-S liquidus is the ideal form on FeS units below 18 GPa, on Fe$_3$S units from 26 GPa and their mean at 22 GPa, that the Fe-C liquidus below 18 GPa is the Fe$_3$C form written out by hand, that a mixture of S, C and O takes the S units of its pressure, that Fe$_3$S units hold the eutectic from 17.83 GPa, that the S and C blends do not overlap and the joins sit at the starts of the tables, and that the gradient of alloys below 30 GPa passes `jax.debug_nans`.

## Limits

The partition of light elements between solid and liquid, the energy budget of an inner core with an alloy liquidus, and a coupled run are outside these cases; they belong to the core energy budget ([core](core.md)).
