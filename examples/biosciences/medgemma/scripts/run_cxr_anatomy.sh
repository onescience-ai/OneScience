#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXAMPLE_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)
source "${REPO_ROOT}/env.sh"
cd $SCRIPT_DIR
pwd


##  single image
HIP_VISIBLE_DEVICES=0 \
	python ./notebook_conver/cxr_anatomy_localization_with_hugging_face.py    \
	--model_path ${ONESCIENCE_DATASETS_DIR}/medgemma/modelscope/google/medgemma-1.5-4b-it  \
     	--image_path "${ONESCIENCE_DATASETS_DIR}/medgemma/Chest_Xray/COVID19_Pneumonia_Normal_Chest_Xray_PA_Dataset/covid/COVID-19 (89).jpg" \
	--object_name "right clavicle" \
	--num_gpus 2

## multiple images
HIP_VISIBLE_DEVICES=0 \
	python ./notebook_conver/cxr_anatomy_localization_with_hugging_face.py \
    	--model_path ${ONESCIENCE_DATASETS_DIR}/medgemma/modelscope/google/medgemma-1.5-4b-it \
    	--input_dir "${ONESCIENCE_DATASETS_DIR}/medgemma/test_images"   \
   	--object_name "right clavicle" \
	--num_gpus 2
