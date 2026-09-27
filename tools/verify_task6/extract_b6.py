#!/usr/bin/env python3
"""Extract benchmark B6 reference values from Euen et al. (2023).

Parses Tables 5 through 10 from the paper text, extracting:
case, code, mesh, Ra, E, Rayleigh-number definition, heating number, Vrms, mean T, Nu_top, Nu_bottom.
Also outputs b6_sources.csv mapping each value to its source table and page.
"""

import csv
import math
import os
import re
from pathlib import Path

BASE_DIR = Path("/Users/timlichtenberg/work/ssc-verify-task6")
DATA_D = BASE_DIR / "data" / "D"
LAYOUT_FILE = DATA_D / "paper_layout.txt"
REF_OUT = DATA_D / "b6_reference.csv"
SRC_OUT = DATA_D / "b6_sources.csv"
REF_WT_OUT = Path(__file__).resolve().parent / "b6_reference.csv"

# Paper parameters:
# Inner radius rb = 0.55, outer radius rt = 1.0 (spherical shell)
# Temperature: T_top = 0, T_bottom = 1
# Viscosity: eta = exp(E * (0.5 - T))
# Rayleigh number Ra = rho * alpha * g * Delta_T * D^3 / (kappa * eta(T=0.5))
# Heating number: 0.0 (purely basally heated, no internal heating in Eqs. 1-3)
CASES = {
    "A1": {
        "Ra": 7000.0,
        "E": 0.0,
        "delta_eta": 1.0,
        "table": 5,
        "page": 6,
        "journal_page": 3226,
    },
    "A3": {
        "Ra": 7000.0,
        "E": math.log(20.0),
        "delta_eta": 20.0,
        "table": 6,
        "page": 8,
        "journal_page": 3228,
    },
    "A7": {
        "Ra": 7000.0,
        "E": math.log(1e5),
        "delta_eta": 1e5,
        "table": 7,
        "page": 10,
        "journal_page": 3230,
    },
    "C1": {
        "Ra": 100000.0,
        "E": 0.0,
        "delta_eta": 1.0,
        "table": 8,
        "page": 16,
        "journal_page": 3236,
    },
    "C2": {
        "Ra": 100000.0,
        "E": math.log(10.0),
        "delta_eta": 10.0,
        "table": 9,
        "page": 16,
        "journal_page": 3236,
    },
    "C3": {
        "Ra": 100000.0,
        "E": math.log(30.0),
        "delta_eta": 30.0,
        "table": 10,
        "page": 17,
        "journal_page": 3237,
    },
}

TABLE_BOUNDARIES = [
    (
        "A1",
        5,
        6,
        r"Table 5\. Results for Case A1[\s\S]*?(?=Figure 3\.|\x0c|\Z)",
    ),
    (
        "A3",
        6,
        8,
        r"Table 6\. Results for Case A3[\s\S]*?(?=amongst|\x0c|\Z)",
    ),
    (
        "A7",
        7,
        10,
        r"Table 7\. Results for Case A7[\s\S]*?(?=Case A3|\x0c|\Z)",
    ),
    (
        "C1",
        8,
        16,
        r"Table 8\. Results for Case C1[\s\S]*?(?=Table 9\.|\x0c|\Z)",
    ),
    (
        "C2",
        9,
        16,
        r"Table 9\. Results for Case C2[\s\S]*?(?=creasing mesh|\x0c|\Z)",
    ),
    (
        "C3",
        10,
        17,
        r"Table 10\. Results for Case C3[\s\S]*?(?=ASPECT use the SUPG|\x0c|\Z)",
    ),
]


def clean_num(s: str) -> float:
    """Parse string number handling unicode minus."""
    s = s.replace("−", "-").strip()
    return float(s)


def parse_tables():
    with open(LAYOUT_FILE, "r", encoding="utf-8") as f:
        full_text = f.read()

    ref_rows = []
    src_rows = []

    for case_id, t_num, page_num, pattern in TABLE_BOUNDARIES:
        c_info = CASES[case_id]
        m = re.search(pattern, full_text)
        if not m:
            raise RuntimeError(f"Could not locate Table {t_num} (Case {case_id})")

        block = m.group(0)
        lines = block.splitlines()

        current_code = None

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            # Check for Code header markers
            if "ASPECT – EV" in line or "ASPECT - EV" in line:
                current_code = "ASPECT-EV"
            elif line_str.startswith("ASPECT"):
                current_code = "ASPECT"
            elif line_str.startswith("CitcomS"):
                # Could be "CitcomS" header or "CitcomS  Zhong et al. (2008)"
                if "Zhong et al." in line_str:
                    current_code = "CitcomS (Zhong et al. 2008)"
                else:
                    current_code = "CitcomS"

            # Check if this line is a data line (has numbers)
            # Match line with: [Mesh] [Vrms] [T] [Nut] [Nub] [% diff]
            # Examples of mesh: "8 radial cells", "16 radial cells", "Extrapolated", "24 radial elements", "Zhong et al. (2008)"
            # A data row contains at least 4 numeric tokens at the end.
            tokens = line_str.split()
            # Find all numeric tokens from the end
            num_tokens = []
            for tok in reversed(tokens):
                tok_clean = tok.replace("−", "-")
                try:
                    float(tok_clean)
                    num_tokens.append(tok_clean)
                except ValueError:
                    break

            if len(num_tokens) >= 4:
                # We have numeric values!
                # Order from end: [% diff (optional)], Nub, Nut, <T>, <Vrms>
                num_tokens.reverse()
                if len(num_tokens) >= 5:
                    vrms_str, t_str, nut_str, nub_str = num_tokens[-5:-1]
                else:
                    # 4 numbers: Vrms, T, Nut, Nub (no % diff)
                    vrms_str, t_str, nut_str, nub_str = num_tokens

                # The mesh text is everything before the numeric tokens
                # But watch out if current_code is on the same line
                prefix = line_str
                # Strip out the numbers
                for tok in num_tokens:
                    # strip from the right
                    idx = prefix.rfind(tok)
                    if idx != -1:
                        prefix = prefix[:idx].strip()

                # Clean up mesh string
                mesh_str = prefix
                for code_name in ["ASPECT – EV", "ASPECT - EV", "ASPECT", "CitcomS"]:
                    if mesh_str.startswith(code_name):
                        mesh_str = mesh_str[len(code_name):].strip()

                if "Zhong et al." in line_str:
                    current_code = "CitcomS (Zhong et al. 2008)"
                    mesh_str = "Zhong et al. (2008)"

                if not mesh_str:
                    # Could be Extrapolated
                    if "Extrapolated" in line_str:
                        mesh_str = "Extrapolated"

                vrms = clean_num(vrms_str)
                mean_t = clean_num(t_str)
                nut = clean_num(nut_str)
                nub = clean_num(nub_str)

                code_val = current_code if current_code else "ASPECT"
                if "Zhong et al." in mesh_str:
                    code_val = "CitcomS (Zhong et al. 2008)"

                row = {
                    "case": case_id,
                    "code": code_val,
                    "mesh": mesh_str,
                    "Ra": f"{c_info['Ra']:.1f}",
                    "E": f"{c_info['E']:.6f}",
                    "Rayleigh-number definition": "viscosity at T = 0.5",
                    "heating number": "0.0",
                    "Vrms": f"{vrms:.4f}",
                    "mean T": f"{mean_t:.6f}",
                    "Nu_top": f"{nut:.5f}",
                    "Nu_bottom": f"{nub:.5f}",
                }
                ref_rows.append(row)

                # Add to sources
                val_id = f"{case_id}_{code_val}_{mesh_str}".replace(" ", "_")
                src_rows.append({
                    "value_id": val_id,
                    "table": f"Table {t_num}",
                    "page": f"Page {page_num} (p. {c_info['journal_page']})",
                })

    print(f"Parsed {len(ref_rows)} rows across {len(TABLE_BOUNDARIES)} tables.")

    # Write b6_reference.csv
    fieldnames = [
        "case",
        "code",
        "mesh",
        "Ra",
        "E",
        "Rayleigh-number definition",
        "heating number",
        "Vrms",
        "mean T",
        "Nu_top",
        "Nu_bottom",
    ]
    for target in [REF_OUT, REF_WT_OUT]:
        with open(target, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(ref_rows)
        print(f"Wrote {len(ref_rows)} rows to {target}")

    # Write b6_sources.csv
    with open(SRC_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["value_id", "table", "page"])
        writer.writeheader()
        writer.writerows(src_rows)
    print(f"Wrote {len(src_rows)} rows to {SRC_OUT}")

    return ref_rows


if __name__ == "__main__":
    parse_tables()
