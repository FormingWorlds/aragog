# Core alloy liquidus

These cases check the liquidus of the core alloy, `aragog.core.AlloyLiquidus` ([explanation](../Explanations/core_alloy_liquidus.md)), against the source papers. Every case is a test in `tests/test_core_alloy_liquidus.py`; the figures are written by those tests.

## 1. End points of each binary

**What is tested.** Each Fe-X binary (S, C, O) runs from the pure-iron melting temperature at zero light element to the eutectic temperature at the eutectic composition, at 10, 60, 135, 254, 330 and 450 GPa, also where the eutectic point is extrapolated.

**Reference.** The construction of the ideal form: $A$ is fixed by the eutectic point.

**Result.** At zero light element the liquidus equals the pure-iron curve exactly; at the eutectic composition it equals the eutectic temperature to a relative $10^{-12}$. Si, H and N leave the liquidus of pure iron unchanged.

**Pinned by** `::test_each_binary_runs_through_pure_iron_and_its_eutectic` and `::test_si_h_and_n_lower_no_melting_point`.

## 2. The Fe-S liquidus curves of Mori et al. (2017)

**What is tested.** The binary form through the end points of each curve that Mori et al. (2017, Fig. 9) draw for the Fe-rich side of the Fe-Fe$_3$S system, at 30, 130 and 250 GPa.

**Reference.** The 3 curves, read twice from the figure (section 6).

**Result.** On Fe$_3$S units the form reproduces the curves to an RMS of 2.06, 2.22 and 0.70 K (first read) and 3.61, 3.27 and 1.37 K (second read) at 30, 130 and 250 GPa; the test bounds are 5 and 6 K. On S atoms the form misses by an RMS of 304, 193 and 136 K, so the figure fixes the composition basis.

![Fe-S liquidus against pressure for 30, 130 and 250 GPa, Aragog lines and the curves of Mori et al. 2017 as circles](../assets/validation/core_alloy_liquidus_fe_s_mori2017.svg#only-light){ width="70%" }
![Fe-S liquidus against pressure for 30, 130 and 250 GPa, Aragog lines and the curves of Mori et al. 2017 as circles](../assets/validation/core_alloy_liquidus_fe_s_mori2017_dark.svg#only-dark){ width="70%" }

**Figure 1.** Liquidus temperature against S content on the Fe-rich side of the eutectic at 30, 130 and 250 GPa. Lines: Aragog, through the end points of each curve; circles: the curves of Mori et al. (2017, Fig. 9), second read.

**What the figure shows.** The circles lie on the lines at all 3 pressures; at 30 GPa the curve bends down towards the eutectic, which the Fe$_3$S units give and S atoms do not.

**Pinned by** `::test_the_fe_s_liquidus_curves_of_mori_are_the_ideal_form_on_fe3s_units`.

## 3. The Fe-C liquidus curves of Mashino et al. (2019)

**What is tested.** The binary form on Fe$_3$C units (54, 138, 203 GPa) and Fe$_7$C$_3$ units (255 GPa and the dashed 330 GPa curve) against the curves of Mashino et al. (2019, Fig. 8). The C units are not verified; the case states the misfit and keeps it from growing.

**Reference.** The 5 curves, read twice from the figure (section 6).

**Result.** The form misses the curves by an RMS of 49.6, 112.3, 121.6, 111.7 and 86.5 K at 54, 138, 203, 255 and 330 GPa (first read); straight lines between the end points miss by 1.1, 41.2, 43.4, 81.7 and 94.4 K. The test holds each RMS to 1 K above these values.

![Fe-C liquidus against C content for 5 pressures, Aragog lines and the curves of Mashino et al. 2019 as circles](../assets/validation/core_alloy_liquidus_fe_c_mashino2019.svg#only-light){ width="70%" }
![Fe-C liquidus against C content for 5 pressures, Aragog lines and the curves of Mashino et al. 2019 as circles](../assets/validation/core_alloy_liquidus_fe_c_mashino2019_dark.svg#only-dark){ width="70%" }

**Figure 2.** Liquidus temperature against C content on the Fe-rich side of the eutectic at 54 to 330 GPa. Lines: Aragog, through the end points of each curve; circles: the curves of Mashino et al. (2019, Fig. 8), first read.

**What the figure shows.** Between the end points the circles lie below the lines: the drawn curves are nearly straight, and the ideal form lies above them by at most 70, 156, 174, 156 and 125 K at 54, 138, 203, 255 and 330 GPa.

**Limits.** Between pure iron and the eutectic the liquidus of a C-bearing alloy differs from the drawn curves by an RMS of 50 to 122 K.

**Pinned by** `::test_the_fe_c_misfit_against_mashino_stays_as_stated`.

## 4. Eutectic temperatures

**What is tested.** The eutectic temperatures against the values the papers state.

**Reference.** Fe-S 3550 K at 254 GPa (Mori et al. 2017, run 11); Fe-C about 3750 K at 330 GPa (Mashino et al. 2019, p. 140); Fe-O on the curve of Oka et al. (2019, Fig. 4).

**Result.** 3541.1 K, 3713.3 K and 3854.8 K, within the 50 K bound for the 2 fits and on the read curve for Fe-O.

**Pinned by** `::test_the_eutectic_temperatures_meet_the_papers`.

## 5. Several light elements

**What is tested.** The harmonic rule for an alloy with 2 wt% S, 0.5 wt% C and 3 wt% O at 135 and 330 GPa, against the same rule and 3 others computed by hand in the test, and its dilute limit.

**Reference.** The rules written out in the test: harmonic $A$, composition-weighted $A$, the sum of the binary depressions and the dilute sum $T_\mathrm{Fe} \sum_i x_i / A_i$.

**Result.** The depression is 640.7 K at 135 GPa and 1800.5 K at 330 GPa (the test bound is 1 K around 641 and 1800 K). The composition-weighted rule gives 488.7 and 892.4 K, the sum of the binaries 664.2 and 2062.4 K, the dilute sum 679.6 and 2300.2 K. Scaled down by 10, 100 and 1000, the depression over the dilute sum is 0.99462, 0.99947 and 0.99995.

![Melting point depression against total light-element content at 135 GPa for the 4 rules](../assets/validation/core_alloy_liquidus_rules_135gpa.svg#only-light){ width="70%" }
![Melting point depression against total light-element content at 135 GPa for the 4 rules](../assets/validation/core_alloy_liquidus_rules_135gpa_dark.svg#only-dark){ width="70%" }

![Melting point depression against total light-element content at 330 GPa for the 4 rules](../assets/validation/core_alloy_liquidus_rules_330gpa.svg#only-light){ width="70%" }
![Melting point depression against total light-element content at 330 GPa for the 4 rules](../assets/validation/core_alloy_liquidus_rules_330gpa_dark.svg#only-dark){ width="70%" }

**Figure 3.** Melting point depression against the total S + C + O content at a fixed ratio of 4 : 1 : 6 by mass, at 135 GPa (top) and 330 GPa (bottom), for the harmonic rule of Aragog and 3 other rules.

**What the figure shows.** All rules agree at small content; the harmonic rule stays close to the sum of the binaries and far from the composition-weighted rule, which halves the depression at 330 GPa.

**Pinned by** `::test_the_harmonic_rule_gives_the_plan_table` and `::test_the_harmonic_rule_meets_the_dilute_limit`.

## 6. The 2 reads of each figure curve

**What is tested.** Every curve taken from a figure (the eutectic compositions of Fe-S, Fe-C, Fe-Si and Fe-O, the Fe-O eutectic temperature and the 8 liquidus curves) was read twice: from the vector path of the curve in the PDF, and from a 300 dpi raster image. The 2 reads must agree.

**Reference.** The first read; the tolerance is half the drawn line width in points of the source figure (the full width for the thin Fe-O temperature line).

**Result.** The largest distance between the reads is 0.07 to 1.35 pt, inside the tolerance of every file (0.35 to 2.34 pt). The second read of the Fe-O composition has no points from 42 to 52 GPa, where an open symbol in the colour of the curve lies on it.

**Pinned by** `::test_the_two_reads_of_each_figure_curve_agree`; the reads are in `tests/validation/data/core_eutectics/`.

## Limits

The partition of light elements between solid and liquid, the energy budget of an inner core with an alloy liquidus, and a coupled run are outside these cases; they belong to the core energy budget ([core](core.md)).
