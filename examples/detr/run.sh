#!/bin/bash
set -euo pipefail
IFS=$'\n\t'
RESULT="result.png"

if test -f "$RESULT"; then
    echo "$RESULT exists. removing.."
    rm -v result.png
fi

echo "Exporting DETR model..."
python export_detr.py

if [ $# -eq 0 ]; then # no argument
    D_ARG=()
    I_ARG=()
else
    D_ARG=("--target" $1 "--device" $2)
    I_ARG=("--target" $1 "--device" $2) # simulatorbf16, sakura2 for example
fi

echo "running deploy..."
python deploy.py "${D_ARG[@]}"

echo "inferencing..."
python demo_model.py "${I_ARG[@]}"

test -f "$RESULT"
echo "passed."
