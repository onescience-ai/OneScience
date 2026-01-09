export ONESCIENCE_DATASETS_DIR="/public/onestore/onedatasets"
export ONESCIENCE_MODELS_DIR="/public/onestore/onemodels"

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