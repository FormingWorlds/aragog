"""Run the given verification items, then item 13, in one process; print every item 13 value."""
import json, sys, tempfile
from pathlib import Path
sys.path[:0] = ['tools/verification', 'tests', '.']
import run_core_verification as script
script.OUT = Path(tempfile.mkdtemp())
before = [int(i) for i in (sys.argv[1] if len(sys.argv) > 1 else '').split(',') if i]
for item in before + [13]:
    script.VALUES[str(item)] = {}
    script.ITEMS[item]()
print('ITEM13_VALUES', ','.join(map(str, before)) or 'alone', json.dumps(script.VALUES['13'], sort_keys=True, default=float), flush=True)
