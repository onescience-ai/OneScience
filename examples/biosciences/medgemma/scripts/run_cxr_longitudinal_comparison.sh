#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXAMPLE_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)
source "${REPO_ROOT}/env.sh"
cd $SCRIPT_DIR
pwd

HIP_VISIBLE_DEVICES=0 \
	python ./notebook_conver/cxr_longitudinal_comparison.py    \
	--model_path ${ONESCIENCE_DATASETS_DIR}/medgemma/modelscope/google/medgemma-1.5-4b-it  \
	--image1 ${ONESCIENCE_DATASETS_DIR}/medgemma/test_compare/longitudinal_cxr_before.png \
	--image2 ${ONESCIENCE_DATASETS_DIR}/medgemma/test_compare/longitudinal_cxr_after.png \
	--output_dir ./compare_outputs \
	#--preprocess   # 可选，如果图像不是正方形且需要填充

