"""The top basic node: linear extrapolation and the limited node, monotone in the top cell."""

from __future__ import annotations

import logging
import tomllib
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st

import aragog.mesh
from aragog.config import Config
from aragog.config.boundary import BoundaryConfig
from aragog.mesh import Mesh, top_node_entropy
from aragog.parser import _BoundaryConditionsParameters

from .test_entropy_solver_helpers import _build_minimal_solver

pytestmark = pytest.mark.unit

_BC = dict(
    outer_boundary_condition=4, outer_boundary_value=0.0, inner_boundary_condition=1,
    inner_boundary_value=0.0, emissivity=1.0, equilibrium_temperature=255.0,
    core_heat_capacity=880.0,
)  # fmt: skip
_BUILDERS = pytest.mark.parametrize(
    'build',
    [
        lambda **kw: _normalized(_BoundaryConditionsParameters(**_BC, **kw)),
        lambda **kw: BoundaryConfig(**_BC, **kw),
    ],
    ids=['_BoundaryConditionsParameters', 'BoundaryConfig'],
)
C, P = 0.5, 4  # scale of the edge distance and power of the smooth minimum


def _normalized(bc):
    bc.normalize()
    return bc


def _linear(S, r):
    return (1 + r / 2) * S[-1] - r / 2 * S[-2]


def _ref(S, r, lo=-np.inf, hi=np.inf):
    """The limited node from its definition, computed here independently of the function."""
    D = 0.5 * r * (S[-1] - S[-2])
    if D == 0.0:
        return S[-1]
    a, b = abs(D), C * max((S[-1] - lo) if D < 0 else (hi - S[-1]), 0.0)
    h = max(a, b)  # scaled so a^p and b^p do not underflow
    M = a if np.isinf(b) else (a / h) * b / ((a / h) ** P + (b / h) ** P) ** (1 / P)
    return S[-1] + np.sign(D) * M


@st.composite
def _table_states(draw, margin=0.0):
    """Edges of a table and the top 3 cells, with the top cell at least ``margin`` inside."""
    lo = draw(st.floats(-500.0, 3000.0))
    span = draw(st.floats(max(1.0, 3 * margin), 30000.0))
    x = draw(st.floats(margin / span, 1.0 - margin / span))
    below, extra = draw(st.floats(-0.2, 1.2)), draw(st.floats(-0.2, 1.2))
    S = np.array([lo + extra * span, lo + below * span, lo + x * span])
    return S, draw(st.floats(0.9, 1.1)), lo, lo + span


_S = st.floats(-5000.0, 30000.0)
_R = st.floats(0.9, 1.1)


@_BUILDERS
@pytest.mark.parametrize('mode', ['linear', 'limited'])
def test_the_top_node_modes_are_accepted(build, mode):
    assert build(top_node=mode).top_node == mode


@_BUILDERS
def test_the_default_top_node_is_linear(build):
    assert build().top_node == 'linear'


@_BUILDERS
def test_the_gradient_core_refuses_the_limited_node(build):
    with pytest.raises(ValueError, match='not available on core_bc'):
        build(core_bc='gradient', top_node='limited')


@_BUILDERS
@given(value=st.sampled_from([True, False, None, 1, 0.5, 'Limited', 'minmod', '', np.nan]))
def test_another_top_node_value_is_refused(build, value):
    """A TOML true, a number, None, another spelling or another word is refused by name."""
    with pytest.raises(ValueError, match='unknown top_node'):
        build(top_node=value)


@pytest.mark.parametrize('first', ['top_node', 'core_bc'])
def test_an_assignment_after_construction_is_checked_and_leaves_the_config(first):
    """'limited' on the gradient core is refused in either order of assignment, and the refused
    assignment leaves the value it found."""
    bc = BoundaryConfig(**_BC)
    setattr(bc, first, 'limited' if first == 'top_node' else 'gradient')
    second, value = ('core_bc', 'gradient') if first == 'top_node' else ('top_node', 'limited')
    kept = getattr(bc, second)
    with pytest.raises(ValueError, match='not available on core_bc'):
        setattr(bc, second, value)
    assert getattr(bc, second) == kept


@given(value=st.sampled_from([True, 1, None, 'Limited']))
def test_another_value_assigned_after_construction_is_refused(value):
    bc = BoundaryConfig(**_BC)
    with pytest.raises(ValueError, match='unknown top_node'):
        bc.top_node = value
    assert bc.top_node == 'linear'


@pytest.mark.parametrize('first', ['top_node', 'core_bc'])
def test_the_mesh_refuses_a_parameter_object_changed_to_limited_on_the_gradient_core(first):
    """The parser dataclass is not checked on assignment; the mesh checks it at construction."""
    s = _build_minimal_solver()
    bc = s.parameters.boundary_conditions
    for name in (first, 'core_bc' if first == 'top_node' else 'top_node'):
        setattr(bc, name, 'limited' if name == 'top_node' else 'gradient')
    with pytest.raises(ValueError, match='not available on core_bc'):
        Mesh(s.parameters)


@pytest.mark.parametrize(
    ('core_bc', 'refused'), [('energy_balance', False), ('gradient', True)]
)
def test_the_toml_reader_takes_the_limited_node_and_refuses_it_on_the_gradient_core(
    core_bc, refused
):
    path = Path(aragog.mesh.__file__).parents[1] / 'cfg' / 'abe_solid.toml'
    with path.open('rb') as f:
        data = tomllib.load(f)
    data['boundary_conditions'] |= {'top_node': 'limited', 'core_bc': core_bc}
    if refused:
        with pytest.raises(ValueError, match='not available on core_bc'):
            Config.from_dict(data, config_dir=path.parent)
    else:
        assert Config.from_dict(data, config_dir=path.parent).boundary_conditions.top_node == (
            'limited'
        )


@given(state=_table_states())
@example(state=(np.array([2500.0, 2000.0, 2000.0]), 1.0, 100.0, 3000.0))  # no step
@example(state=(np.array([2500.0, 2000.0, 100.0]), 1.0, 100.0, 3000.0))  # top cell at S_min
@example(state=(np.array([150.0, 200.0, 3000.0]), 1.0, 100.0, 3000.0))  # top cell at S_max
def test_the_node_is_the_smooth_minimum_of_the_half_step_and_half_the_edge_distance(state):
    S, r, lo, hi = state
    assert top_node_entropy(S, r, lo, hi) == pytest.approx(
        _ref(S, r, lo, hi), rel=1e-12, abs=1e-9
    )


def test_a_cold_top_cell_takes_the_value_of_the_definition():
    """Top step -600 at r = 1: half step a = 300, half the distance to S_min = 100 is b = 150,
    so the node is 400 - 300 * 150 / (300^4 + 150^4)^(1/4)."""
    S = np.array([1300.0, 1000.0, 400.0])
    want = 400.0 - 45000.0 / 8.60625e9**0.25
    assert top_node_entropy(S, 1.0, 100.0, 3000.0) == pytest.approx(want, rel=1e-14)
    assert 252.2 < want < 252.3


@given(state=_table_states())
def test_with_the_top_cell_in_the_table_the_node_keeps_half_its_distance_to_each_edge(state):
    S, r, lo, hi = state
    node, tol = top_node_entropy(S, r, lo, hi), 1e-9 * (abs(lo) + abs(hi))
    assert lo + (1 - C) * (S[-1] - lo) - tol <= node <= hi - (1 - C) * (hi - S[-1]) + tol


@given(s=st.tuples(_S, _S, _S), r=_R, lo=_S, span=st.floats(0.0, 30000.0))
def test_the_node_lies_between_the_top_cell_and_the_linear_node(s, r, lo, span):
    S = np.array(s)
    node, lin = top_node_entropy(S, r, lo, lo + span), _linear(S, r)
    assert min(S[-1], lin) - 1e-9 <= node <= max(S[-1], lin) + 1e-9


def _slopes(S, r, lo, hi, h=1e-3):
    """Central differences of the node in the top cell and in the cell below it."""
    out = []
    for i in (-1, -2):
        up, dn = S.copy(), S.copy()
        up[i] += h
        dn[i] -= h
        out.append(
            (top_node_entropy(up, r, lo, hi) - top_node_entropy(dn, r, lo, hi)) / (2 * h)
        )
    return out


@given(state=_table_states(margin=1.0))
# steep cold top near S_min
@example(state=(np.array([1106.1, 1806.1, 455.7]), 1.0, 106.1, 3000.0))
def test_the_node_rises_with_the_top_cell_and_falls_with_the_cell_below(state):
    """d node / d S_N >= 1 - C and -r/2 <= d node / d S_N-1 <= 0 inside the table."""
    S, r, lo, hi = state
    d_top, d_below = _slopes(S, r, lo, hi)
    assert d_top >= 1 - C - 1e-5
    assert -r / 2 - 1e-5 <= d_below <= 1e-5


@pytest.mark.parametrize('edge', ['S_min', 'S_max'])
def test_at_a_table_edge_the_slope_is_one_outside_and_one_minus_c_inside(edge):
    """The node is the top cell outside the table and keeps 1 - C of the distance inside."""
    lo, hi, h = 100.0, 3000.0, 1e-3
    S = (
        np.array([lo + 2500.0, lo + 2000.0, lo])
        if edge == 'S_min'
        else np.array([hi - 2500.0, hi - 2000.0, hi])
    )
    sgn = 1.0 if edge == 'S_min' else -1.0
    node = lambda x: top_node_entropy(np.array([*S[:2], S[2] + sgn * x]), 1.0, lo, hi)  # noqa: E731
    assert (node(0.0) - node(-h)) / (sgn * h) == pytest.approx(1.0, rel=1e-9)
    assert (node(h) - node(0.0)) / (sgn * h) == pytest.approx(1 - C, rel=1e-6)


@pytest.mark.parametrize(('ratio', 'gap'), [(4.0, 9.742e-4), (2.0, 1.504e-2), (1.0, 0.1591)])
def test_the_node_returns_to_linear_as_the_edge_moves_away(ratio, gap):
    """With b = ratio * a the node lies 1 - (1 + ratio^-4)^(-1/4) of the half step off linear."""
    a = 300.0
    S = np.array([1300.0, 1000.0, 1000.0 - 2 * a])
    lo = S[-1] - ratio * a / C
    rel = (top_node_entropy(S, 1.0, lo, 1e5) - _linear(S, 1.0)) / a
    assert rel == pytest.approx(1 - (1 + ratio**-P) ** (-1 / P), rel=1e-12)
    assert rel == pytest.approx(gap, rel=1e-3)


@pytest.mark.parametrize('bounds', [(-np.inf, np.inf), (100.0, np.inf), (2e3, np.inf)])
def test_a_zero_step_on_an_unbounded_side_raises_no_warning(bounds):
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert top_node_entropy(np.array([1e3, 2e3, 2e3]), 1.0, *bounds) == 2e3


@given(s=st.tuples(_S, _S, _S), r=_R)
def test_an_unbounded_table_gives_the_linear_node(s, r):
    S = np.array(s)
    assert top_node_entropy(S, r) == pytest.approx(_linear(S, r), rel=1e-12, abs=1e-9)


@pytest.mark.parametrize(
    ('S', 'side'), [([400.0, 300.0, 50.0], 'below'), ([2700.0, 2800.0, 3050.0], 'above')]
)
def test_a_top_cell_outside_the_table_stepping_away_from_it_is_its_own_node(S, side):
    assert top_node_entropy(np.array(S), 1.0, 100.0, 3000.0) == S[-1]


@pytest.mark.parametrize(
    'S', [[1.0, 20.0, 50.0], [1.0, 4000.0, 3100.0]], ids=['below', 'above']
)
def test_a_top_cell_outside_the_table_stepping_into_it_takes_the_far_edge(S):
    """The half step points into the table, so b is half the distance to the far edge."""
    S = np.array(S)
    node = top_node_entropy(S, 1.0, 100.0, 3000.0)
    assert node == pytest.approx(_ref(S, 1.0, 100.0, 3000.0), rel=1e-12) and node != S[-1]


def test_the_node_is_continuous_across_a_zero_step():
    """The cell below moves through the top cell in steps of 1e-3: the node moves by about the
    half step, on both sides of the zero step and next to S_min."""
    below = 160.0 + 1e-3 * np.arange(-20000, 20001)
    cells = [np.full_like(below, 180.0), below, np.full_like(below, 160.0)]
    nodes = top_node_entropy(np.vstack(cells), 1.0, 100.0, 3000.0)
    assert np.max(np.abs(np.diff(nodes))) < 0.6e-3


@given(S=st.tuples(_S, _S, _S), r=_R, bad=st.sampled_from([np.nan, np.inf, -np.inf]))
def test_a_non_finite_cell_gives_a_non_finite_node(S, r, bad):
    for i in (1, 2):
        cells = np.array(S)
        cells[i] = bad
        assert not np.isfinite(top_node_entropy(cells, r, -6000.0, 30000.0))


@pytest.mark.parametrize(
    'cells', [[2e3, np.inf], [2e3, -np.inf], [np.inf, 2e3], [-np.inf, 2e3]]
)
def test_a_non_finite_cell_on_a_bounded_table_gives_nan_without_a_warning(cells):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        node = top_node_entropy(np.array(cells), 1.0, 100.0, 3000.0)
    assert np.isnan(node)
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]


def test_two_cells_give_the_node_and_one_cell_is_refused():
    assert np.isfinite(top_node_entropy(np.array([2000.0, 400.0]), 1.0, 100.0, 3000.0))
    with pytest.raises(ValueError, match='at least 2 cells'):
        top_node_entropy(np.array([400.0]), 1.0, 100.0, 3000.0)


def test_the_jax_node_matches_numpy_and_its_jacobian_matches_differences():
    jax = pytest.importorskip('jax')
    jnp = jax.numpy
    jax.config.update('jax_enable_x64', True)
    rng = np.random.default_rng(3)
    lo, hi = 106.1, 3000.0
    cells = np.column_stack(
        [
            rng.uniform(lo, hi, 400),
            rng.uniform(lo, hi, 400),
            rng.uniform(lo + 2.0, hi - 2.0, 400),
        ]
    )
    f = jax.jit(lambda S: top_node_entropy(S, 1.03, lo, hi, xp=jnp))
    jac = jax.jit(jax.jacrev(f))
    for S in cells:
        assert float(f(jnp.asarray(S))) == pytest.approx(
            top_node_entropy(S, 1.03, lo, hi), rel=1e-13
        )
        assert np.array(jac(jnp.asarray(S)))[:0:-1] == pytest.approx(
            _slopes(S, 1.03, lo, hi, h=1e-4), rel=1e-5, abs=1e-6
        )
    for b_lo, b_hi in ((lo, hi), (-np.inf, np.inf)):  # a zero step has the linear slopes
        zero = jax.jacrev(lambda S: top_node_entropy(S, 1.03, b_lo, b_hi, xp=jnp))
        assert np.array(zero(jnp.asarray([1e3, 2e3, 2e3]))) == pytest.approx(
            [0.0, -0.515, 1.515]
        )
    for S in ([0.0, 900.0, 500.0], [0.0, -2000.0, 500.0]):  # exact ties a == b at both edges
        tie = jax.jacrev(lambda x: top_node_entropy(x, 1.0, 100.0, 3000.0, xp=jnp))
        fd = _slopes(np.array(S), 1.0, 100.0, 3000.0, h=1e-4)
        assert np.array(tie(jnp.asarray(S)))[:0:-1] == pytest.approx(fd, rel=1e-5, abs=1e-6)
    edge = jax.jacrev(lambda x: top_node_entropy(x, 1.0, 100.0, 3000.0, xp=jnp))
    for S in ([0.0, 500.0, 100.0], [0.0, 2500.0, 3000.0]):  # top cell on the edge it steps to
        assert float(np.array(edge(jnp.asarray(S)))[-1]) == pytest.approx(1 - C)
    assert np.isnan(top_node_entropy(np.array([1e3, 2e3, 500.0]), 1.0, np.nan, 3000.0))
    for S in ([1e3, 2e3, lo], [lo, lo, lo]):  # top cell at S_min
        assert np.all(np.isfinite(np.array(jac(jnp.asarray(S)))))
    with pytest.raises(ValueError, match='at least 2 cells'):
        top_node_entropy(jnp.asarray([400.0]), 1.0, lo, hi, xp=jnp)


def _limited_mesh(bounds=(100.0, 3000.0)):
    s = _build_minimal_solver()
    s.parameters.boundary_conditions.top_node = 'limited'
    mesh = Mesh(s.parameters)
    mesh.top_bounds = bounds
    return mesh


def test_the_mesh_puts_the_limited_node_on_top_of_the_quantity_transform():
    mesh = _limited_mesh()
    S = np.linspace(2900.0, 200.0, mesh._quantity_transform.shape[1])
    basic, plain = mesh.entropy_at_basic_nodes(S), mesh.quantity_at_basic_nodes(S).ravel()
    assert np.array_equal(basic[:-1], plain[:-1])
    assert basic[-1] == top_node_entropy(S, mesh.top_ratio, 100.0, 3000.0) != plain[-1]
    mesh.top_node = 'linear'
    assert np.array_equal(mesh.entropy_at_basic_nodes(S), plain)


def test_the_mesh_refuses_a_node_that_breaks_its_bounds(monkeypatch):
    """A node outside the table, or not between the top cell and the linear node, stops the
    call: the check runs on every state the mesh evaluates."""
    mesh = _limited_mesh()
    S = np.linspace(2900.0, 200.0, mesh._quantity_transform.shape[1])
    for bad in (99.0, 210.0):
        monkeypatch.setattr(aragog.mesh, 'top_node_entropy', lambda *a, v=bad, **k: v)
        with pytest.raises(RuntimeError, match='limited top node'):
            mesh.entropy_at_basic_nodes(S)
    # the linear node is in the table: only the tolerance
    wide = _limited_mesh((-1e4, 1e4))
    lin = wide.quantity_at_basic_nodes(S).ravel()[-1]
    monkeypatch.setattr(aragog.mesh, 'top_node_entropy', lambda *a, **k: lin * (1 - 1e-10))
    with pytest.raises(RuntimeError, match='limited top node'):
        wide.entropy_at_basic_nodes(S)
    monkeypatch.setattr(aragog.mesh, 'top_node_entropy', lambda *a, **k: lin * (1 - 1e-13))
    assert wide.entropy_at_basic_nodes(S)[-1] == lin * (1 - 1e-13)


def test_the_jax_mesh_carries_the_mode_and_bounds_and_gives_the_numpy_node():
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    from aragog.jax.phase import MeshArrays, entropy_at_basic_nodes

    mesh = _limited_mesh((150.0, 2950.0))
    S = np.linspace(2900.0, 160.0, mesh._quantity_transform.shape[1])
    arrays = MeshArrays.from_numpy_mesh(mesh)
    assert arrays.top_limited and tuple(np.array(arrays.top_bounds)) == (150.0, 2950.0)
    assert np.array(entropy_at_basic_nodes(arrays, jax.numpy.asarray(S))) == pytest.approx(
        mesh.entropy_at_basic_nodes(S), rel=1e-13
    )


def test_the_floor_margin_does_not_enter_the_node():
    """A margin of NaN, inf or 0 assigned after construction leaves the node and its bounds."""
    s = _build_minimal_solver()
    s.parameters.boundary_conditions.top_node = 'limited'
    s._initialize_internals()
    s.entropy_eos = SimpleNamespace(S_min=100.0, S_max=3000.0)
    s._sync_top_node()
    mesh = s.evaluator.mesh
    S = np.linspace(2900.0, 200.0, mesh._quantity_transform.shape[1])
    want = mesh.entropy_at_basic_nodes(S)
    assert want[-1] != mesh.quantity_at_basic_nodes(S).ravel()[-1]
    for margin in (np.nan, np.inf, 0.0, 150.0):
        s.parameters.energy.eos_floor_margin = margin
        s._sync_top_node()
        assert mesh.top_bounds == (100.0, 3000.0)
        assert np.array_equal(mesh.entropy_at_basic_nodes(S), want)


@pytest.mark.parametrize(
    'bounds', [None, (100.0, 3000.0), (100.0, np.inf)], ids=['no-eos', 'eos', 'half-open']
)
def test_the_limited_mode_log_names_the_bounds_or_warns_without_a_table(caplog, bounds):
    """Without EOS table bounds the limited node equals the linear node: the solver warns; with
    bounds, also one-sided ones, it names them; a second initialize logs nothing new."""
    s = _build_minimal_solver()
    s.parameters.boundary_conditions.top_node = 'limited'
    if bounds:
        s.entropy_eos = SimpleNamespace(S_min=bounds[0], S_max=bounds[1])
        s._check_eos_floors = lambda: None
    with caplog.at_level(logging.INFO, logger='fwl.aragog'):
        s.initialize()
        s.initialize()
    rec = [r for r in caplog.records if 'top node: limited' in r.getMessage()]
    assert len(rec) == 1
    if bounds:
        want = f'[{bounds[0]:g}, {bounds[1]:g}] J/kg/K'
        assert rec[0].levelno == logging.INFO and want in rec[0].getMessage()
    else:
        assert (
            rec[0].levelno == logging.WARNING
            and 'equals the linear node' in rec[0].getMessage()
        )


def test_the_limited_mode_log_follows_a_new_eos(caplog):
    s = _build_minimal_solver()
    s.parameters.boundary_conditions.top_node = 'limited'
    s.entropy_eos = SimpleNamespace(S_min=100.0, S_max=3000.0)
    s._check_eos_floors = lambda: None
    s.initialize()
    s.entropy_eos = SimpleNamespace(S_min=25.99, S_max=3000.0)
    caplog.clear()  # an earlier test can leave the logger at INFO
    with caplog.at_level(logging.INFO, logger='fwl.aragog'):
        s._sync_top_node()
    rec = [r.getMessage() for r in caplog.records if 'top node: limited' in r.getMessage()]
    assert rec == [
        'EntropySolver: top node: limited (monotone, inside the table [25.99, 3000] J/kg/K)'
    ]
    assert s.evaluator.mesh.top_bounds == (25.99, 3000.0)


@pytest.mark.parametrize('bounds', [(np.nan, 3000.0), (3000.0, 100.0), (100.0, 100.0)])
def test_a_limited_node_refuses_a_table_range_that_is_not_an_interval(bounds):
    s = _build_minimal_solver()
    s.parameters.boundary_conditions.top_node = 'limited'
    s._initialize_internals()
    s.entropy_eos = SimpleNamespace(S_min=bounds[0], S_max=bounds[1])
    with pytest.raises(ValueError, match='not an interval'):
        s._sync_top_node()
    assert s.evaluator.mesh.top_bounds == (-np.inf, np.inf)


def test_the_linear_mode_reads_no_table_range_of_the_eos():
    """In linear mode any object can stand as the EOS: the bounds sync does not read it."""
    s = _build_minimal_solver()
    s._initialize_internals()
    s.entropy_eos = object()
    s._sync_top_node()
    mesh = s.evaluator.mesh
    assert mesh.top_node == 'linear' and mesh.top_bounds == (-np.inf, np.inf)


def test_the_limited_node_is_the_linear_row_bitwise_where_the_limiter_does_not_act():
    """Far from the table edges the limited node is the outer row of the quantity transform bit
    for bit, in numpy and in JAX, so limited and linear runs agree to the last bit there."""
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    from aragog.jax.phase import MeshArrays, entropy_at_basic_nodes

    mesh = _limited_mesh((-1.0e6, 1.0e7))
    arrays = MeshArrays.from_numpy_mesh(mesh)
    rng, n = np.random.default_rng(5), mesh._quantity_transform.shape[1]
    differ = [0, 0]
    for _ in range(500):
        S = rng.uniform(2000.0, 4000.0) + rng.normal(0.0, 1.0, n)
        differ[0] += mesh.entropy_at_basic_nodes(S)[-1] != mesh.quantity_at_basic_nodes(S)[-1]
        Sj = jax.numpy.asarray(S)
        differ[1] += float(entropy_at_basic_nodes(arrays, Sj)[-1]) != float(
            (arrays.quantity_matrix @ Sj)[-1]
        )
    assert differ == [0, 0]
