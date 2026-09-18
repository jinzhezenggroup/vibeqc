#!/usr/bin/env bash
set -euo pipefail
root=/inspire/qb-ilm/project/chemicalreaction/diwenxi-CZXS25120072/vibeqc-workspace
evidence="$root/build-171/fprojector-20260918/evidence"
while [ ! -f "$evidence/baseline.exit" ]; do sleep 10; done
exec bash "$root/evidence-171/fprojector-run-cuda.sh" cuda
