"""Standalone core-evolution module and the config-driven budget factory.

Two couplings expose the same physics. The solver-coupled path appends
``T_cmb`` to the entropy solver's state vector (``core_bc =
'core_module'``), so the core temperature advances inside every CVODE
sub-step. The module here is the pure coupling: :class:`CoreModule` holds
the core state itself and advances it over an externally supplied heat-flow
interval with fixed Runge-Kutta sub-steps, recording the sub-step
trajectory so a caller integrating over a coupling step never loses the
fast transients inside it. Both couplings share :class:`CoreEnergyBudget`,
so they can be compared on identical physics.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from aragog.core.budget import CoreEnergyBudget
from aragog.core.entropy import CoreEntropyBudget
from aragog.core.melting import IronMeltingCurve, QuadraticMeltingCurve
from aragog.core.profiles import GaussianCoreProfiles
from aragog.core.regime import crystallization_regime

jax.config.update('jax_enable_x64', True)

# Factory defaults: Earth-like values so a bare configuration is runnable.
# The entropy of fusion is the PALEOS calibration anchor, 1.16 k_B per atom
# of iron (Zhang et al. 2015), converted to 172.8 J/kg/K.
_FACTORY_DEFAULTS = {
    'rho_cen': 12500.0,
    'length_scale': 7272e3,
    'alpha': 1.35e-5,
    'c_p': 840.0,
    'ds_fusion': 172.8,
    'icn_width': 10.0,
}


def build_core_module_budget(
    params: dict, *, r_cmb: float, p_cmb_fallback: float
) -> CoreEnergyBudget:
    """Build a :class:`CoreEnergyBudget` from a flat config dict.

    Recognised keys: profile parameters (``rho_cen``, ``length_scale``,
    ``p_cmb``, ``alpha``, ``c_p``, ``pressure_mode``), the melting-curve
    selector ``melting_curve`` (``'iron'`` with ``light_element_fraction``
    and ``depression``, or ``'quadratic'`` with ``t_m0``/``t_m1``/``t_m2``),
    and the budget parameters (``ds_fusion``, ``icn_width``,
    ``latent_heat``, ``alpha_c``, ``c_light``, ``capacity_mode``,
    ``legacy_rho_core``, ``legacy_tfac``). The CMB radius comes from the
    caller (the solver's mesh), never from the dict, and ``p_cmb`` falls
    back to the caller's value when absent.

    Raises
    ------
    ValueError
        From the underlying constructors on any invalid value, or here on
        an unknown melting-curve selector or unrecognised key.
    """
    params = {**_FACTORY_DEFAULTS, **params}
    profile_keys = {'rho_cen', 'length_scale', 'p_cmb', 'alpha', 'c_p', 'pressure_mode'}
    curve_kind = params.pop('melting_curve', 'iron')
    curve_keys = {
        'iron': {'light_element_fraction', 'depression'},
        'quadratic': {'t_m0', 't_m1', 't_m2'},
    }
    if curve_kind not in curve_keys:
        raise ValueError(f'unknown melting_curve {curve_kind!r}')
    budget_keys = {
        'ds_fusion',
        'icn_width',
        'latent_heat',
        'alpha_c',
        'c_light',
        'capacity_mode',
        'legacy_rho_core',
        'legacy_tfac',
    }
    # Both curves' keys are recognised regardless of the active selector:
    # a config surface (the PROTEUS attrs block) carries every field, and
    # only the active curve's subset is consumed.
    known = profile_keys | curve_keys['iron'] | curve_keys['quadratic'] | budget_keys
    unknown = set(params) - known
    if unknown:
        raise ValueError(f'unrecognised core_module_params keys: {sorted(unknown)}')

    profile_kwargs = {k: params[k] for k in profile_keys if k in params}
    profile_kwargs['r_cmb'] = r_cmb
    profile_kwargs.setdefault('p_cmb', p_cmb_fallback)
    profiles = GaussianCoreProfiles(**profile_kwargs)

    curve_kwargs = {k: params[k] for k in curve_keys[curve_kind] if k in params}
    if curve_kind == 'iron':
        curve = IronMeltingCurve(**curve_kwargs)
    else:
        curve = QuadraticMeltingCurve(**curve_kwargs)

    budget_kwargs = {k: params[k] for k in budget_keys if k in params}
    return CoreEnergyBudget(profiles, curve, **budget_kwargs)


class CoreModule:
    """Stateful core evolving under an externally supplied CMB heat flow.

    Parameters
    ----------
    budget : CoreEnergyBudget
        The energy budget that supplies the cooling rate.
    t_cmb : float
        Initial CMB temperature [K], positive.
    entropy : CoreEntropyBudget, optional
        When given, diagnostics include the entropy margin and field
        strength alongside the energy-side quantities.
    q_radio : float
        Internal source power [W] passed to every cooling-rate call.
    n_substeps : int
        Fixed classical Runge-Kutta sub-steps per :meth:`step` call;
        positive. The trajectory arrays carry ``n_substeps + 1`` samples.
    """

    def __init__(
        self,
        budget: CoreEnergyBudget,
        *,
        t_cmb: float,
        entropy: CoreEntropyBudget | None = None,
        q_radio: float = 0.0,
        n_substeps: int = 32,
    ) -> None:
        if not float(t_cmb) > 0.0:
            raise ValueError(f't_cmb must be positive, got {t_cmb}')
        if int(n_substeps) < 1:
            raise ValueError(f'n_substeps must be at least 1, got {n_substeps}')
        self.budget = budget
        self.entropy = entropy
        self.t_cmb = float(t_cmb)
        self.q_radio = float(q_radio)
        self.n_substeps = int(n_substeps)
        self.last_times: jnp.ndarray | None = None
        self.last_t_cmb: jnp.ndarray | None = None
        self.last_q_cmb: jnp.ndarray | None = None
        # One compiled RHS per instance: the RK4 loop calls it hundreds of
        # times per step, and the eager path re-traces the boundary
        # bisection on every call. q_radio is a traced argument, not a
        # closure capture: jit freezes closed-over Python values at first
        # trace, so a later mutation of the attribute would silently keep
        # the old power.
        self._rhs = jax.jit(
            lambda temp, q, q_radio: self.budget.dtcmb_dt(temp, q, q_sources=q_radio)
        )

    def step(self, q_cmb: float, dt: float, q_cmb_end: float | None = None) -> float:
        """Advance ``T_cmb`` over ``dt`` seconds of heat flow ``q_cmb`` [W].

        The heat flow ramps linearly from ``q_cmb`` to ``q_cmb_end`` (equal
        to ``q_cmb`` when omitted) across the interval, and the state
        advances with ``n_substeps`` classical RK4 sub-steps. The sub-step
        trajectory is stored on ``last_times`` / ``last_t_cmb`` /
        ``last_q_cmb`` so the caller can see through the coupling interval
        instead of only its endpoints. Returns the new ``T_cmb`` [K].

        Raises
        ------
        ValueError
            If ``dt`` is not positive.
        """
        if not float(dt) > 0.0:
            raise ValueError(f'dt must be positive, got {dt}')
        q_end = float(q_cmb) if q_cmb_end is None else float(q_cmb_end)
        h = float(dt) / self.n_substeps

        def q_at(t: float) -> float:
            return float(q_cmb) + (q_end - float(q_cmb)) * (t / float(dt))

        def rhs(t: float, temp):
            return self._rhs(temp, q_at(t), self.q_radio)

        times = [0.0]
        temps = [self.t_cmb]
        temp = jnp.asarray(self.t_cmb)
        t = 0.0
        for _ in range(self.n_substeps):
            k1 = rhs(t, temp)
            k2 = rhs(t + h / 2.0, temp + h / 2.0 * k1)
            k3 = rhs(t + h / 2.0, temp + h / 2.0 * k2)
            k4 = rhs(t + h, temp + h * k3)
            temp = temp + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            t += h
            times.append(t)
            temps.append(float(temp))
        self.last_times = jnp.asarray(times)
        self.last_t_cmb = jnp.asarray(temps)
        self.last_q_cmb = jnp.asarray([q_at(x) for x in times])
        self.t_cmb = float(temp)
        return self.t_cmb

    def diagnostics(self, q_cmb: float | None = None) -> dict:
        """Current core state as plain floats.

        Always contains ``t_cmb``, ``t_cen``, ``r_icb``,
        ``nucleation_factor``, ``effective_capacity``, and the phase flags
        ``inner_core_present`` / ``fully_frozen``. With an entropy budget
        attached and ``q_cmb`` given, adds ``entropy_margin``,
        ``dynamo_active``, and ``b_rms_core``.
        """
        b = self.budget
        t = self.t_cmb
        radius = float(b.r_icb(t))
        out = {
            't_cmb': t,
            't_cen': float(b.profiles.t_cen(t)),
            'r_icb': radius,
            'nucleation_factor': float(b.nucleation_factor(t)),
            'effective_capacity': float(b.effective_capacity(t)),
            'inner_core_present': bool(radius > 0.0),
            'fully_frozen': bool(~b._liquid_remains(t)),
            'regime': int(crystallization_regime(b, t)),
        }
        if self.entropy is not None and q_cmb is not None:
            margin = float(self.entropy.entropy_margin(t, q_cmb, self.q_radio))
            out['entropy_margin'] = margin
            out['dynamo_active'] = bool(margin > 0.0)
            out['b_rms_core'] = float(self.entropy.b_rms_core(t, q_cmb))
        return out
