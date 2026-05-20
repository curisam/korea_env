#!/usr/bin/env bash
# Build a reproducible conda environment for FedBiscuit_ultra.
# Usage: bash setup.sh [env_name]
set -euo pipefail

ENV_NAME="${1:-fedbiscuit}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found. Install Miniconda/Anaconda first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "[1/3] Creating conda env: ${ENV_NAME} (Python 3.9)"
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "      env '${ENV_NAME}' already exists; skipping create"
else
  conda create -y -n "${ENV_NAME}" python=3.9 pip
fi
conda activate "${ENV_NAME}"

echo "[2/3] Installing pip packages (torch cu121 + HF stack + FS deps)"
pip install --upgrade pip
pip install -r requirements.txt

echo "[3/3] Installing FederatedScope in editable mode"
pip install -e .

echo
echo "===== Environment '${ENV_NAME}' ready ====="
python - <<'PY'
import torch
print(f"  python   : {__import__('sys').version.split()[0]}")
print(f"  torch    : {torch.__version__}")
print(f"  cuda ok  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device 0 : {torch.cuda.get_device_name(0)}")
import transformers, accelerate, peft, datasets
print(f"  transformers : {transformers.__version__}")
print(f"  accelerate   : {accelerate.__version__}")
print(f"  peft         : {peft.__version__}")
print(f"  datasets     : {datasets.__version__}")
PY
