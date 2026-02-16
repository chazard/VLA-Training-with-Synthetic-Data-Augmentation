#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "[setup] Initializing submodules..."
git submodule sync --recursive
git submodule update --init --recursive

# Pick one: uv (fast) or venv+pip (more universal)
USE_UV="${USE_UV:-0}"

if [[ "$USE_UV" == "1" ]]; then
  if ! command -v uv >/dev/null 2>&1; then
    echo "[setup] uv not found. Install uv or run with USE_UV=0"
    exit 1
  fi
  echo "[setup] Creating uv environment + installing deps..."
  uv venv .venv
  source .venv/bin/activate
  uv pip install -U pip
  # Install your own package deps (expects pyproject.toml)
  uv pip install -e ".[dev]"
else
  echo "[setup] Creating venv..."
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install -U pip wheel setuptools
  # Install your own package deps (expects pyproject.toml)
  python -m pip install -e ".[dev]"
fi

echo "[setup] Done."
echo "Next:"
echo "  source .venv/bin/activate"
echo "  ./scripts/print_pins.sh"
