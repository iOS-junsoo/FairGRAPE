#!/usr/bin/env bash
set -euo pipefail
# Latest user correction: the A full run has 22 rounds, preserving 5 epochs/round.
exec bash /workspace/FairGRAPE/FairGRAPE/scripts/run_celeba_A_full22.sh "$@"
