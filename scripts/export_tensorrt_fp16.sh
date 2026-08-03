#!/usr/bin/env bash
set -euo pipefail

WEIGHTS="${1:-weights/aquayolo26n_seaclear_seed42.pt}"
python aquayolo26/export.py --weights "$WEIGHTS" --format engine --imgsz 640 --half --device 0
