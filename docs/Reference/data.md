# Reference Data

## Download instructions

Aragog requires lookup table data for thermophysical properties of liquid and solid silicate. These data are stored in a Zenodo/OSF repository and can be downloaded with:

```console
aragog download all
```

The command `aragog env` shows the path where data will be downloaded. To set a custom path:

```console
export FWL_DATA=/your/absolute/path/
aragog download all
```

## Data requirements

Aragog needs the following types of lookup data, depending on the configuration:

### Phase property lookup tables

When `[phase_liquid]` or `[phase_solid]` properties are specified as file paths instead of constants, Aragog loads pressure-dependent (1D) or pressure-temperature-dependent (2D) lookup tables.

| Format | Columns | Usage |
|--------|---------|-------|
| 1D lookup | pressure, property value | Single-property P-dependence |
| 2D lookup | pressure, temperature, property value | Full P-T dependence |

### Solidus and liquidus curves

The `[phase_mixed]` section specifies solidus and liquidus curves as file paths. These are pressure-temperature lookup tables:

| Column | Unit | Description |
|--------|------|-------------|
| pressure | Pa | Pressure values |
| temperature | K | Melting/freezing temperature at that pressure |

### External EOS tables (PROTEUS integration)

When Aragog is called from PROTEUS with external EOS data (e.g. PALEOS tables), the EOS tables are passed through the configuration dictionary via `eos_method = 2`. In this mode, pressure, density, and gravity profiles are provided by the calling code (e.g. Zalmoxis structure solver) rather than computed internally by Aragog's Adams-Williamson EOS.

## Adams-Williamson EOS (built-in)

When `eos_method = 1` (the default), Aragog uses a built-in Adams-Williamson pressure profile with constant gravitational acceleration:

$$
\rho^*(P) = \rho^*_{\mathrm{top}} \exp\!\left(\frac{P}{B}\right),
\qquad
\frac{dP}{dr} = -\rho^*(P)\, g.
$$

This requires no external data. Parameters are set in the `[mesh]` section:

- `surface_density`: $\rho^*_{\mathrm{top}}$ in kg/m^3
- `adiabatic_bulk_modulus`: $B$ in Pa
- `gravitational_acceleration`: $g$ in m/s^2
