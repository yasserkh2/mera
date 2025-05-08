#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

if [ $# -eq 0 ]; then # no argument
    D_ARG=()
    I_ARG=()
else
    D_ARG=("--target" $1 "--device" $2)
    I_ARG=("--target" $1 "--device" $2) # simulatorbf16, sakura2 for example
fi

echo "Quantizing model ..."
python quantize_model.py

echo "Deploying model ..."
python deploy.py "${D_ARG[@]}"

echo "Inferencing model ..."
python demo_model.py "${I_ARG[@]}"

echo "passed."
