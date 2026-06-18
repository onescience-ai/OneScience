export ONESCIENCE_DATASETS_DIR="/public/share/sugonhpcapp01/onestore/onedatasets"
export ONESCIENCE_MODELS_DIR="/public/share/sugonhpcapp01/onestore/onemodels"
export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
export device="gpu" # gpu or dcu

# check datasets path
if [ ! -d "$ONESCIENCE_DATASETS_DIR" ]; then
  echo "❌️❌️ ERROR: ONESCIENCE_DATASETS_DIR does not exist:"
  echo "    $ONESCIENCE_DATASETS_DIR"
fi

# check models path
if [ ! -d "$ONESCIENCE_MODELS_DIR" ]; then
  echo "❌❌️ ERROR: ONESCIENCE_MODELS_DIR does not exist:"
  echo "    $ONESCIENCE_MODELS_DIR"
fi

echo "✅️✅️ Device Variables Are Set: ${device}"
