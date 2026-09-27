#!/bin/bash
R=~/.shared-agent-state/ssc-verify-task6
D=~/work/ssc-verify-task6/data
miss=0
need() { [ -s "$1" ] || { echo "MISSING or EMPTY: $1"; miss=$((miss+1)); }; }
for f in REPORT.md CHECKPOINT.md claims.csv findings.csv mutations.csv tips.txt; do need "$R/$f"; done
for f in S/testdata_sha256.txt S/base_aragog_main.log S/import_canary.log \
         A/A1_summary.csv A/A2_discrimination.csv A/A3_compare.json A/A4_bitwise.csv \
         A/A5_summary.csv A/A6.json \
         B/B1_halfspace.csv B/B2.json B/B3_slab.csv B/B4.json B/B5_parity.csv \
         D/b6_reference.csv D/b6_mapping.md D/b6_run.py D/b6_runs.csv MANIFEST.tsv; do need "$D/$f"; done
for n in 1 2 3 4 5 6 7 8; do need "$D/C/C$n/result.csv"; done
head -1 "$R/claims.csv" | grep -q '^id,claim,source,dev2_value,task6_value,unit,tolerance,status,evidence_path$' || { echo "BAD HEADER: claims.csv"; miss=$((miss+1)); }
if grep -qvE ',(REPRODUCED|MISMATCH|NOT RUN),[^,]*$|^id,' "$R/claims.csv"; then echo "BAD STATUS in claims.csv"; miss=$((miss+1)); fi
du -sk "$D" | awk '{ if ($1 > 5*1024*1024) { print "DATA OVER 5 GB"; exit 1 } }' || miss=$((miss+1))
[ "$miss" -eq 0 ] && echo "DELIVERABLES COMPLETE" || { echo "$miss problem(s)"; exit 1; }
