#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="${ROOT_DIR}/.venv/bin/python"

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "Missing ${VENV_PYTHON}. Create the repo virtualenv first." >&2
  exit 1
fi

cd "${ROOT_DIR}"
exec "${VENV_PYTHON}" -m trading.bot "$@"
