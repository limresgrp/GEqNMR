#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ENV="${ROOT_DIR}/.venv-geqnmr"
DEFAULT_PY="3.11"

prompt() {
  local label="$1"
  local default="$2"
  local value
  read -r -p "${label} [${default}]: " value
  if [ -z "$value" ]; then
    echo "$default"
  else
    echo "$value"
  fi
}

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.cargo/bin:$PATH"
fi

PY_VER="$(prompt "Python version" "${DEFAULT_PY}")"
VENV_DIR="$(prompt "Virtual env directory" "${DEFAULT_ENV}")"

if [ -d "${VENV_DIR}" ]; then
  read -r -p "Remove existing environment at ${VENV_DIR}? [y/N]: " remove_env
  if [[ "${remove_env}" =~ ^[Yy]$ ]]; then
    rm -rf "${VENV_DIR}"
  fi
fi

if [ ! -d "${VENV_DIR}" ]; then
  if ! uv venv "${VENV_DIR}" --python "${PY_VER}" --seed; then
    uv python install "${PY_VER}"
    uv venv "${VENV_DIR}" --python "${PY_VER}" --seed
  fi
fi

source "${VENV_DIR}/bin/activate"
uv pip install -r "${ROOT_DIR}/backend/requirements.txt"

echo "Environment ready."
echo "Activate with: source ${VENV_DIR}/bin/activate"
