
set -euo pipefail
IFS=$'\n\t'

rm -f result*.png


if [ $# -eq 0 ]; then # no argument
    D_ARG=()
    I_ARG=()
else
    D_ARG=("--target" $1 "--device" $2)
    I_ARG=("--target" $1 "--device" $2) # simulatorbf16, sakura2 for example
fi

echo "exporting ..."
./export_onnx.sh

echo "quantizing model ..."
python quantize_model.py \
  --model_path="./source_model_files/yolov8m_hardswish_finetuned_640x640.onnx" \
  --coco_dir="./data/calib_images" \
  --calib_imgs_num=5


echo "deploying model ..."
python deploy.py "${D_ARG[@]}"

echo "inferencing..."
python demo_model.py "${I_ARG[@]}"

if ls result*.png 1> /dev/null 2>&1; then
    echo "passed."
    # TODO: Make sure the content is also valid and correct
else
    echo "failed. No output was generated"
    exit 1
fi
