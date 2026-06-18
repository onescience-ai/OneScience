#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="esm2_t36_3B_UR50D"
ONEMODELS_DIR="${ONESCIENCE_MODELS_DIR:-/public/share/sugonhpcapp01/onestore/onemodels}"

ESM_REPO_SRC=""
WEIGHT_SRC=""
CONTACT_SRC=""
SIMPLEFOLD_DIR=""
SIMPLEFOLD_CACHE_DIR=""
SIMPLEFOLD_MODEL_CKPT=""
PLDDT_CKPT=""
CCD_SRC=""
BOLTZ_CONF_SRC=""
FULL_VALIDATE=0
USE_CONTACT_REGRESSION=0

usage() {
  cat <<USAGE
Usage:
  bash setup_esm_torchhub.sh [options]

Purpose:
  Prepare offline assets required by the SimpleFold validation stage in the
  protein-design-to-structure-validation workflow.

Default model root:
  ${ONEMODELS_DIR}

Default layout under model root:
  esm_models/esm-main
  esm_models/${MODEL_NAME}.pt
  esm_models/${MODEL_NAME}-contact-regression.pt
  simplefold/simplefold_100M.ckpt
  simplefold/plddt.ckpt
  simplefold/ccd.pkl
  simplefold/boltz1_conf.ckpt

Options:
  --onemodels-dir DIR    Override OneModels root.
  --esm-repo-src PATH    ESM source repo directory containing hubconf.py.
  --weight PATH          ${MODEL_NAME}.pt source file.
  --contact PATH         ${MODEL_NAME}-contact-regression.pt source file.
  --simplefold-dir DIR   SimpleFold checkpoint directory.
  --simplefold-model-ckpt PATH
                         SimpleFold model checkpoint used by --simplefold_model.
                         Default: simplefold/simplefold_100M.ckpt.
  --plddt-ckpt PATH      pLDDT checkpoint used when SimpleFold runs with --plddt.
                         Default: simplefold/plddt.ckpt.
  --simplefold-cache DIR Directory for ccd.pkl and boltz1_conf.ckpt.
                         Default: --simplefold-dir.
  --ccd PATH             Optional source ccd.pkl copied into --simplefold-cache.
  --boltz-conf PATH      Optional source boltz1_conf.ckpt copied into --simplefold-cache.
  --use-contact          Keep official ESM behavior and require/load the
                         contact-regression checkpoint. By default this script
                         patches hubconf.py to skip contact-regression because
                         SimpleFold only uses ESM representations.
  --full-validate        Run torch.hub.load after copying. This may take memory
                         and time for the 3B model.
  -h, --help             Show this help.

USAGE
}

log() {
  printf '[setup-esm] %s\n' "$*"
}

die() {
  printf '[setup-esm][ERROR] %s\n' "$*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --onemodels-dir)
      ONEMODELS_DIR="${2:-}"
      shift 2
      ;;
    --esm-repo-src)
      ESM_REPO_SRC="${2:-}"
      shift 2
      ;;
    --weight)
      WEIGHT_SRC="${2:-}"
      shift 2
      ;;
    --contact)
      CONTACT_SRC="${2:-}"
      shift 2
      ;;
    --simplefold-dir)
      SIMPLEFOLD_DIR="${2:-}"
      shift 2
      ;;
    --simplefold-model-ckpt)
      SIMPLEFOLD_MODEL_CKPT="${2:-}"
      shift 2
      ;;
    --plddt-ckpt)
      PLDDT_CKPT="${2:-}"
      shift 2
      ;;
    --simplefold-cache)
      SIMPLEFOLD_CACHE_DIR="${2:-}"
      shift 2
      ;;
    --ccd)
      CCD_SRC="${2:-}"
      shift 2
      ;;
    --boltz-conf)
      BOLTZ_CONF_SRC="${2:-}"
      shift 2
      ;;
    --use-contact)
      USE_CONTACT_REGRESSION=1
      shift
      ;;
    --full-validate)
      FULL_VALIDATE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

[[ -n "$ONEMODELS_DIR" ]] || die "ONEMODELS_DIR is empty."

ESM_BASE="${ONEMODELS_DIR}/esm_models"
ESM_REPO_SRC="${ESM_REPO_SRC:-${ESM_BASE}/esm-main}"
WEIGHT_SRC="${WEIGHT_SRC:-${ESM_BASE}/${MODEL_NAME}.pt}"
CONTACT_SRC="${CONTACT_SRC:-${ESM_BASE}/${MODEL_NAME}-contact-regression.pt}"
SIMPLEFOLD_DIR="${SIMPLEFOLD_DIR:-${ONEMODELS_DIR}/simplefold}"
SIMPLEFOLD_CACHE_DIR="${SIMPLEFOLD_CACHE_DIR:-${SIMPLEFOLD_DIR}}"
SIMPLEFOLD_MODEL_CKPT="${SIMPLEFOLD_MODEL_CKPT:-${SIMPLEFOLD_DIR}/simplefold_100M.ckpt}"
PLDDT_CKPT="${PLDDT_CKPT:-${SIMPLEFOLD_DIR}/plddt.ckpt}"

command -v python >/dev/null 2>&1 || die "python not found. Activate the target conda env first."

[[ -d "$ESM_REPO_SRC" ]] || die "ESM repo directory not found: $ESM_REPO_SRC"
[[ -f "${ESM_REPO_SRC}/hubconf.py" ]] || die "hubconf.py not found in ESM repo directory: $ESM_REPO_SRC"
[[ -d "${ESM_REPO_SRC}/esm" ]] || die "esm Python package directory not found under: $ESM_REPO_SRC"
[[ -f "$WEIGHT_SRC" ]] || die "ESM2 weight not found: $WEIGHT_SRC"
if [[ "$USE_CONTACT_REGRESSION" -eq 1 ]]; then
  [[ -f "$CONTACT_SRC" ]] || die "Contact regression weight not found: $CONTACT_SRC"
elif [[ ! -f "$CONTACT_SRC" ]]; then
  log "Contact regression weight not found, continuing because --use-contact is not set:"
  log "  ${CONTACT_SRC}"
fi

validate_simplefold_checkpoints() {
  [[ -d "$SIMPLEFOLD_DIR" ]] || die "SimpleFold checkpoint directory not found: $SIMPLEFOLD_DIR"
  [[ -f "$SIMPLEFOLD_MODEL_CKPT" ]] || die "SimpleFold model checkpoint not found: $SIMPLEFOLD_MODEL_CKPT"
  [[ -f "$PLDDT_CKPT" ]] || die "SimpleFold pLDDT checkpoint not found: $PLDDT_CKPT"

  log "SimpleFold checkpoints are ready:"
  ls -lh "$SIMPLEFOLD_MODEL_CKPT" "$PLDDT_CKPT"
}

install_simplefold_cache() {
  mkdir -p "$SIMPLEFOLD_CACHE_DIR"

  local ccd_dest="${SIMPLEFOLD_CACHE_DIR}/ccd.pkl"
  local boltz_dest="${SIMPLEFOLD_CACHE_DIR}/boltz1_conf.ckpt"

  if [[ -n "$CCD_SRC" ]]; then
    [[ -f "$CCD_SRC" ]] || die "ccd.pkl source not found: $CCD_SRC"
    log "Copying SimpleFold ccd.pkl:"
    log "  ${CCD_SRC} -> ${ccd_dest}"
    cp -f "$CCD_SRC" "$ccd_dest"
  fi

  if [[ -n "$BOLTZ_CONF_SRC" ]]; then
    [[ -f "$BOLTZ_CONF_SRC" ]] || die "boltz1_conf.ckpt source not found: $BOLTZ_CONF_SRC"
    log "Copying SimpleFold boltz1_conf.ckpt:"
    log "  ${BOLTZ_CONF_SRC} -> ${boltz_dest}"
    cp -f "$BOLTZ_CONF_SRC" "$boltz_dest"
  fi

  [[ -f "$ccd_dest" ]] || die "Missing SimpleFold offline cache file: $ccd_dest"
  [[ -f "$boltz_dest" ]] || die "Missing SimpleFold offline cache file: $boltz_dest"

  log "SimpleFold offline cache is ready:"
  ls -lh "$ccd_dest" "$boltz_dest"
}

validate_simplefold_checkpoints
install_simplefold_cache

HUB_DIR="$(python - <<'PY'
import torch
print(torch.hub.get_dir())
PY
)"

[[ -n "$HUB_DIR" ]] || die "torch.hub.get_dir() returned an empty path."

case "$HUB_DIR" in
  */.cache/torch/hub|*/torch/hub)
    ;;
  *)
    die "Refusing to clear unexpected torch hub directory: $HUB_DIR"
    ;;
esac

REPO_DEST="${HUB_DIR}/facebookresearch_esm_main"
CHECKPOINT_DIR="${HUB_DIR}/checkpoints"
WEIGHT_DEST="${CHECKPOINT_DIR}/${MODEL_NAME}.pt"
CONTACT_DEST="${CHECKPOINT_DIR}/${MODEL_NAME}-contact-regression.pt"

log "torch hub dir: ${HUB_DIR}"
log "Clearing torch hub cache contents under: ${HUB_DIR}"
mkdir -p "$HUB_DIR"
find "$HUB_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +

log "Creating torch hub directories"
mkdir -p "$REPO_DEST" "$CHECKPOINT_DIR"

log "Copying ESM repo:"
log "  ${ESM_REPO_SRC} -> ${REPO_DEST}"
cp -a "${ESM_REPO_SRC}/." "$REPO_DEST/"

log "Copying model weight:"
log "  ${WEIGHT_SRC} -> ${WEIGHT_DEST}"
cp -f "$WEIGHT_SRC" "$WEIGHT_DEST"

if [[ -f "$CONTACT_SRC" ]]; then
  log "Copying contact regression weight:"
  log "  ${CONTACT_SRC} -> ${CONTACT_DEST}"
  cp -f "$CONTACT_SRC" "$CONTACT_DEST"
else
  log "Skipping contact regression weight copy."
fi

validate_contact_checkpoint() {
  python - "$CONTACT_DEST" <<'PY'
import pathlib
import sys
import torch

path = pathlib.Path(sys.argv[1])
obj = torch.load(path, map_location="cpu")
state = obj.get("model", obj) if isinstance(obj, dict) else {}
required = {
    "contact_head.regression.weight",
    "contact_head.regression.bias",
}
missing = required - set(state)
if missing:
    preview = sorted(str(k) for k in list(state)[:20])
    raise SystemExit(
        "Contact regression checkpoint is not compatible. "
        f"Missing keys: {sorted(missing)}. "
        f"First keys in checkpoint: {preview}"
    )
print(f"OK: contact regression checkpoint contains {sorted(required)}")
PY
}

patch_hubconf_skip_contact() {
  local hubconf="${REPO_DEST}/hubconf.py"
  log "Patching hubconf.py to load ${MODEL_NAME} without contact-regression"
  cat >> "$hubconf" <<'PY'

# SimpleFold offline override:
# SimpleFold uses ESM representations only, not contact prediction. Loading the
# contact regression head is therefore unnecessary and can fail when the cached
# contact-regression file is absent or from an incompatible ESM release.
def esm2_t36_3B_UR50D():
    from pathlib import Path
    import torch
    from esm.pretrained import load_model_and_alphabet_core

    model_path = Path(torch.hub.get_dir()) / "checkpoints" / "esm2_t36_3B_UR50D.pt"
    model_data = torch.load(str(model_path), map_location="cpu")
    return load_model_and_alphabet_core("esm2_t36_3B_UR50D", model_data, None)
PY
}

if [[ "$USE_CONTACT_REGRESSION" -eq 1 ]]; then
  validate_contact_checkpoint
else
  patch_hubconf_skip_contact
fi

log "Checking hubconf callable without loading the 3B model"
python - "$REPO_DEST" "$MODEL_NAME" <<'PY'
import importlib.util
import pathlib
import sys

repo_dir = pathlib.Path(sys.argv[1])
model_name = sys.argv[2]
hubconf = repo_dir / "hubconf.py"

repo_dir_str = str(repo_dir)
sys.path = [p for p in sys.path if p != repo_dir_str]
sys.path.insert(0, repo_dir_str)
for name in list(sys.modules):
    if name == "esm" or name.startswith("esm."):
        del sys.modules[name]

spec = importlib.util.spec_from_file_location("esm_hubconf_check", hubconf)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

if not hasattr(module, model_name):
    raise SystemExit(f"hubconf.py exists, but callable {model_name!r} is missing")

print(f"OK: found callable {model_name} in {hubconf}")
PY

if [[ "$FULL_VALIDATE" -eq 1 ]]; then
  log "Running full torch.hub.load validation. This may take a while."
  python - "$MODEL_NAME" "$REPO_DEST" <<'PY'
import pathlib
import sys
import torch

model_name = sys.argv[1]
repo_dir = pathlib.Path(sys.argv[2])
repo_dir_str = str(repo_dir)
sys.path = [p for p in sys.path if p != repo_dir_str]
sys.path.insert(0, repo_dir_str)
for name in list(sys.modules):
    if name == "esm" or name.startswith("esm."):
        del sys.modules[name]

try:
    model, alphabet = torch.hub.load(
        repo_dir_str,
        model_name,
        source="local",
        trust_repo=True,
    )
except TypeError:
    model, alphabet = torch.hub.load(
        repo_dir_str,
        model_name,
        source="local",
    )

print(f"OK: torch.hub.load loaded {model_name}")
print(f"model type: {type(model)}")
print(f"alphabet size: {len(alphabet)}")
PY
else
  log "Skipping full torch.hub.load validation. Use --full-validate to test model loading."
fi

log "Done."
