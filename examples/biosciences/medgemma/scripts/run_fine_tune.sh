#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXAMPLE_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)
source "${REPO_ROOT}/env.sh"
cd $SCRIPT_DIR
pwd


# ==========================================
# 自动检查并升级关键依赖
# ==========================================
echo "🔍  自动检查并修复关键依赖版本..."
python -c "
import pkg_resources
pkg_resources.require('transformers==5.12.1')
" || pip install --upgrade transformers==5.12.1

echo "🔍  自动检查 accelerate 版本..."
python -c "
import pkg_resources
pkg_resources.require('accelerate>=0.29.0')
" || pip install --upgrade accelerate
# 如果升级到最新版（如 1.15.x+）出现循环导入报错，可将上面那行换成：
# " || pip install accelerate==1.0.0

# ==========================================
# 执行微调训练
# ==========================================
HIP_VISIBLE_DEVICES=0 \
	python ./notebook_conver/fine_tune_with_hugging_face.py    \
	--model_path ${ONESCIENCE_DATASETS_DIR}/medgemma/modelscope/google/medgemma-1.5-4b-it  \
	--train_zip ${ONESCIENCE_DATASETS_DIR}/medgemma/nct/NCT-CRC-HE-100K.zip \
	--test_zip ${ONESCIENCE_DATASETS_DIR}/medgemma/nct/CRC-VAL-HE-7K.zip \
	--output_dir ./medgemma-nct-lora \
	--max_train_samples 9000 --max_val_samples 1000 --max_test_samples 1000

