"""Fallback of ``EntropySolver.solve`` when a registered JAX CVODE factory raises."""

from __future__ import annotations

import pytest

from aragog.solver.cvode_jax import UnsupportedCoreBCError
from tests.conftest import EOS_DIR, entropy_eos_copy, needs_eos, option_z_parameters, returns

pytest.importorskip('scikits_odes_sundials', reason='scikits_odes_sundials missing')

pytestmark = [pytest.mark.unit, needs_eos]


@pytest.mark.parametrize('err', [UnsupportedCoreBCError, ValueError, RuntimeError])
def test_only_an_unsupported_core_bc_falls_back_to_the_fd_jacobian(err, caplog):
    """An UnsupportedCoreBCError from the factory gives the NumPy RHS with a warning; a
    plain ValueError or a RuntimeError (JAX missing) stops the solve."""
    import logging

    from aragog.solver.entropy_solver import EntropySolver

    solver = EntropySolver(option_z_parameters(), entropy_eos=entropy_eos_copy(EOS_DIR))
    solver.initialize()
    solver.set_initial_entropy(3050.0)

    def factory(scales, core_bc_mode):
        raise err('factory refused')

    solver.set_jax_cvode_factory(factory)
    with caplog.at_level(logging.WARNING, logger='fwl.aragog.solver.entropy_solver'):
        if err is not UnsupportedCoreBCError:
            with pytest.raises(Exception, match='factory refused') as exc_info:
                solver.solve()
            assert exc_info.type is err
            return
        returns(solver.solve)
    assert solver._solution.status == 0 and 'factory refused' in caplog.text
