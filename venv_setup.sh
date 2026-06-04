#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEFAULT_PYTHON="3.11"
DEFAULT_VENV_DIR="${ROOT_DIR}/.venv-geqnmr"

PYTHON_VERSION="${DEFAULT_PYTHON}"
VENV_DIR="${DEFAULT_VENV_DIR}"
TORCH_BACKEND="auto"
TORCH_VERSION=""
INSTALL_TORCH=1
RECREATE_VENV=0
TORCH_BACKEND_SET=0

usage() {
  cat <<USAGE
Usage: ./venv_setup.sh [options]

Create a uv-based GEqNMR virtual environment and install runtime dependencies.

Options:
  --python VERSION         Python version to use (default: ${DEFAULT_PYTHON})
  --venv-dir PATH          Virtual environment directory (default: ${DEFAULT_VENV_DIR})
  --torch-backend BACKEND  uv torch backend: auto|cpu|cu118|cu121|cu124|cu126|cu128|rocm
  --torch-version VERSION  Torch version to install (default: latest)
  --no-torch               Skip torch installation
  --recreate               Remove existing venv before creating a new one
  -h, --help               Show this help message

When --torch-backend is omitted in an interactive shell, the script asks which
CUDA/PyTorch backend to install. Use auto if you want uv to choose.

Examples:
  ./venv_setup.sh --recreate
  ./venv_setup.sh --recreate --torch-backend auto
  ./venv_setup.sh --recreate --torch-backend cu126
  ./venv_setup.sh --recreate --torch-backend cpu
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_VERSION="$2"
      shift 2
      ;;
    --venv-dir)
      VENV_DIR="$2"
      shift 2
      ;;
    --torch-backend)
      TORCH_BACKEND="$2"
      TORCH_BACKEND_SET=1
      shift 2
      ;;
    --torch-version)
      TORCH_VERSION="$2"
      shift 2
      ;;
    --no-torch)
      INSTALL_TORCH=0
      shift
      ;;
    --recreate)
      RECREATE_VENV=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return
  fi

  echo "uv not found. Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh

  if [[ -d "${HOME}/.local/bin" ]]; then
    export PATH="${HOME}/.local/bin:${PATH}"
  fi
  if [[ -d "${HOME}/.cargo/bin" ]]; then
    export PATH="${HOME}/.cargo/bin:${PATH}"
  fi

  if ! command -v uv >/dev/null 2>&1; then
    echo "Failed to find uv after installation. Add ~/.local/bin or ~/.cargo/bin to PATH and retry." >&2
    exit 1
  fi
}

ensure_uv

valid_torch_backend() {
  case "$1" in
    auto|cpu|cu118|cu121|cu124|cu126|cu128|rocm)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

prompt_torch_backend() {
  if [[ "${INSTALL_TORCH}" -ne 1 || "${TORCH_BACKEND_SET}" -eq 1 || ! -t 0 ]]; then
    return
  fi

  cat <<PROMPT

Select the PyTorch backend to install:
  auto   Let uv choose the torch build
  cpu    CPU-only torch
  cu118  CUDA 11.8
  cu121  CUDA 12.1
  cu124  CUDA 12.4
  cu126  CUDA 12.6
  cu128  CUDA 12.8
  rocm   AMD ROCm
PROMPT

  local selected
  while true; do
    read -r -p "Torch backend [cu126]: " selected
    selected="${selected:-cu126}"
    if valid_torch_backend "${selected}"; then
      TORCH_BACKEND="${selected}"
      break
    fi
    echo "Invalid backend: ${selected}" >&2
  done
}

prompt_torch_backend

if ! valid_torch_backend "${TORCH_BACKEND}"; then
  echo "Invalid --torch-backend value: ${TORCH_BACKEND}" >&2
  usage
  exit 1
fi

if [[ "${RECREATE_VENV}" -eq 1 && -d "${VENV_DIR}" ]]; then
  echo "Removing existing venv: ${VENV_DIR}"
  rm -rf "${VENV_DIR}"
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Creating venv at ${VENV_DIR} (Python ${PYTHON_VERSION})"
  if ! uv venv "${VENV_DIR}" --python "cpython-${PYTHON_VERSION}" --seed; then
    uv python install "cpython-${PYTHON_VERSION}"
    uv venv "${VENV_DIR}" --python "cpython-${PYTHON_VERSION}" --seed
  fi
else
  echo "Using existing venv: ${VENV_DIR}"
fi

PYTHON_BIN="${VENV_DIR}/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python binary not found in venv: ${PYTHON_BIN}" >&2
  exit 1
fi

echo "Installing base packaging tools..."
uv pip install --python "${PYTHON_BIN}" -U pip setuptools wheel

if [[ "${INSTALL_TORCH}" -eq 1 ]]; then
  TORCH_SPEC="torch"
  if [[ -n "${TORCH_VERSION}" ]]; then
    TORCH_SPEC="torch==${TORCH_VERSION}"
  fi

  echo "Installing ${TORCH_SPEC} (backend: ${TORCH_BACKEND})"
  if [[ "${TORCH_BACKEND}" == "auto" ]]; then
    uv pip install --python "${PYTHON_BIN}" "${TORCH_SPEC}"
  else
    uv pip install --python "${PYTHON_BIN}" "${TORCH_SPEC}" --torch-backend "${TORCH_BACKEND}"
  fi
fi

echo "Installing GEqNMR runtime dependencies..."
uv pip install --python "${PYTHON_BIN}" -r "${ROOT_DIR}/backend/requirements.txt"

echo
echo "Environment ready."
echo "Activate with:"
echo "  source ${VENV_DIR}/bin/activate"
echo
echo "Running PyTorch/CUDA check..."
"${PYTHON_BIN}" - <<'PY'
import torch

print(f"torch: {torch.__version__}")
print(f"torch.version.cuda: {torch.version.cuda}")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        print(f"cuda:{index}: {torch.cuda.get_device_name(index)}")
else:
    print("CUDA is not available in this environment.")
PY
