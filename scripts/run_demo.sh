#!/usr/bin/env bash
set -euo pipefail

python demo/01_wt_dynamics.py
python demo/02_in_silico_perturbation.py
echo "Demo outputs are in demo/output/"
