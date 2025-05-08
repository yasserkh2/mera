#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# MERA virtual environment
MERA_VENV=mera-env
python3 -m venv $MERA_VENV

echo "source ${MERA_VENV}/bin/activate" > start.sh

echo "---"
echo "To enable the virtual environment please run:"
echo "source start.sh"
echo "---"

