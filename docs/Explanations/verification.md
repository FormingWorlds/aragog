# First-principles verification

Aragog includes a set of verification tests that compare numerical results against known analytical solutions. These tests validate the ODE system, spatial discretization, and boundary condition implementation independently of any physical EOS data.

## Test categories

### Solver convergence tests

Basic checks that the BDF integrator converges for all standard configuration files (solid, liquid, mixed phase, mixed with lookup tables). These tests verify that the solver reaches the end time without failure and produces physically plausible output.

### Phase evaluator tests

Unit tests for the phase evaluator hierarchy (single-phase, mixed-phase, composite). These verify that:

- Property lookups return values in the correct range
- The composite evaluator selects the correct sub-evaluator based on temperature
- Mixed-phase properties transition smoothly between solid and liquid end-members

## Running the verification suite

```console
pytest tests/
```

Or run specific test files:

```console
pytest tests/test_abe.py      # Solver convergence for all config types
pytest tests/test_phase.py    # Phase evaluator correctness
```

## Physical constraints verified

The test suite checks the following physical constraints:

1. **Temperature positivity**: $T > 0$ everywhere at all times
2. **Melt fraction bounds**: $0 \le \phi \le 1$
3. **Flux sign conventions**: conductive flux opposes temperature gradient
4. **Energy conservation**: total energy change matches integrated boundary fluxes plus source terms (within solver tolerance)
5. **Smooth phase transitions**: no spurious oscillations at solidus/liquidus crossings
