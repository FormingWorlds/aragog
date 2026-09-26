#!/usr/bin/env python3
"""Step A6: PROTEUS option off is bitwise.

Verifies:
1. Golden trajectory on wt-proteus-tl_aragog-surface-half-cell vs wt-proteus-main
   has identical values for all existing columns and adds exactly 3 new constant 0.0 columns.
2. Mesh pass-through test on wt-proteus-tl_aragog-mesh-refinement confirms that
   when mesh refinement fields are 0.0, they are not passed to aragog ((None, None)).
"""

import json
import os
import sys
import subprocess
from pathlib import Path

def parse_tsv(lines):
    comments = []
    cols = {}
    for line in lines:
        if line.startswith("#"):
            comments.append(line.strip())
        else:
            parts = line.strip().split("\t")
            if parts and parts[0]:
                cols[parts[0]] = parts[1:]
    return comments, cols

def main():
    root = Path("/Users/timlichtenberg/work/ssc-verify-task6")
    main_tsv = root / "wt-proteus-main/tests/integration/golden_run.tsv"
    branch_tsv = root / "wt-proteus-tl_aragog-surface-half-cell/tests/integration/golden_run.tsv"

    with open(main_tsv, "r") as f:
        main_comm, main_cols = parse_tsv(f.readlines())
    with open(branch_tsv, "r") as f:
        branch_comm, branch_cols = parse_tsv(f.readlines())

    new_cols = sorted(list(set(branch_cols.keys()) - set(main_cols.keys())))
    missing_cols = sorted(list(set(main_cols.keys()) - set(branch_cols.keys())))

    diff_cols = []
    for k in main_cols:
        if main_cols[k] != branch_cols[k]:
            diff_cols.append(k)

    print(f"Main columns: {len(main_cols)}, Branch columns: {len(branch_cols)}")
    print(f"New columns: {new_cols}")
    print(f"Missing columns: {missing_cols}")
    print(f"Diff existing columns: {diff_cols}")

    golden_equal = (len(diff_cols) == 0 and len(missing_cols) == 0 and
                    new_cols == ["G_top_half", "T_top_cell", "w_solid_top"] and
                    all(branch_cols[c] == ["const", "0.0"] for c in new_cols))

    # Mesh pass-through test
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/interior_energetics/test_aragog.py",
        "-k", "test_setup_solver_passes_mesh_refinement_only_when_on",
        "-q", "--tb=short"
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root}/wt-proteus-tl_aragog-mesh-refinement/src:{root}/wt-aragog-tl_aragog-mesh-refinement/src"
    res = subprocess.run(cmd, cwd=root / "wt-proteus-tl_aragog-mesh-refinement", env=env, capture_output=True, text=True)
    mesh_pass = (res.returncode == 0)

    out_json = {
        "status": "DONE",
        "golden_equal": bool(golden_equal),
        "new_columns": new_cols,
        "mesh_passthrough_off": bool(mesh_pass)
    }

    out_json_path = root / "data/A/A6.json"
    with open(out_json_path, "w") as f:
        json.dump(out_json, f, indent=2)

    print("Result JSON:", out_json)

if __name__ == "__main__":
    main()
