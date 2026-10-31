"""Writer of the core verification values file, without the physics imports of the script."""

from __future__ import annotations

import json


def dump_values(values: dict) -> str:
    """The values file text: one top-level item per line, keys sorted, no indentation."""
    rows = (f'{json.dumps(k)}: {json.dumps(values[k], sort_keys=True)}' for k in sorted(values))
    return '{\n' + ',\n'.join(rows) + '\n}\n'
