#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# main whl
pip install mera_visualizer*.whl

echo "-- MERA-VISUALIZER library successfuly installed."
