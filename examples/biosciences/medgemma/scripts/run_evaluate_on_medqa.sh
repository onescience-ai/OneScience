#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXAMPLE_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)
source "${REPO_ROOT}/env.sh"
cd $SCRIPT_DIR
pwd


HIP_VISIBLE_DEVICES=0 \
	python ./notebook_conver/evaluate_on_medqa.py    \
	--model_path ${ONESCIENCE_DATASETS_DIR}/medgemma/modelscope/google/medgemma-1.5-4b-it  \
	--parquet_dir ${ONESCIENCE_DATASETS_DIR}/medgemma/medqa \
	--output_dir ./medqa_results \
	--max_samples 10   # 可选：先测试 100 条

