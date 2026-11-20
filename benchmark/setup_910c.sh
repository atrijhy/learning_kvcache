#!/usr/bin/env bash
# Evaluation dependencies only; follows setup_910c_kvcache.sh's copied-Python setup.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASE_PYTHON=${BASE_PYTHON:-/usr/local/miniconda3/bin/python}
ENV_PREFIX=${ENV_PREFIX:-/usr/local/miniconda3/envs/kvcache-eval}
PIP_INDEX_URL=${PIP_INDEX_URL:-https://mirrors.jd.com/pypi/web/simple}
[[ $(uname -m) == aarch64 ]] || { echo 'Run on the Ascend aarch64 host' >&2; exit 1; }
if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
    [[ ! -e "$ENV_PREFIX" ]] || { echo "Incomplete environment: $ENV_PREFIX" >&2; exit 1; }
    BASE_EXECUTABLE=$(readlink -f -- "$BASE_PYTHON")
    PYTHON_ABI=$($BASE_PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    mkdir -p "$ENV_PREFIX/bin" "$ENV_PREFIX/lib/python$PYTHON_ABI/site-packages"
    install -m 0755 "$BASE_EXECUTABLE" "$ENV_PREFIX/bin/python"
    printf '%s\n' "home = $(dirname -- "$BASE_EXECUTABLE")" \
        'include-system-site-packages = true' "executable = $BASE_EXECUTABLE" > "$ENV_PREFIX/pyvenv.cfg"
fi
[[ -z $(find "$ENV_PREFIX" -type l -print -quit) ]] || { echo 'Symlink found in evaluation environment' >&2; exit 1; }
mkdir -p "$ENV_PREFIX/conda-meta" "$ENV_PREFIX/etc/conda/activate.d"
touch "$ENV_PREFIX/conda-meta/history"
printf '%s\n' '#!/usr/bin/env bash' 'source /usr/local/Ascend/cann-9.1.0/set_env.sh' \
    'export LD_LIBRARY_PATH="/usr/local/miniconda3/lib:/usr/local/miniconda3/lib64:${LD_LIBRARY_PATH:-}"' \
    > "$ENV_PREFIX/etc/conda/activate.d/910c-cann.sh"
"$ENV_PREFIX/bin/python" -m pip install --index-url "$PIP_INDEX_URL" -r "$HERE/requirements-bfcl-910c.txt"
[[ -z $(find "$ENV_PREFIX" -type l -print -quit) ]] || { echo 'Symlink found after installation' >&2; exit 1; }
# The BFCL SDK pins are isolated from native vLLM. Reuse the image stack in the
# project's usual kvcache runtime, installing only the local model registry plugin.
RUNTIME_PREFIX=${KVCACHE_RUNTIME_PREFIX:-/usr/local/miniconda3/envs/kvcache}
if [[ ! -x "$RUNTIME_PREFIX/bin/python" ]]; then
    [[ ! -e "$RUNTIME_PREFIX" ]] || { echo "Incomplete runtime: $RUNTIME_PREFIX" >&2; exit 1; }
    BASE_EXECUTABLE=$(readlink -f -- "$BASE_PYTHON")
    PYTHON_ABI=$($BASE_PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    mkdir -p "$RUNTIME_PREFIX/bin" "$RUNTIME_PREFIX/lib/python$PYTHON_ABI/site-packages"
    install -m 0755 "$BASE_EXECUTABLE" "$RUNTIME_PREFIX/bin/python"
    printf '%s\n' "home = $(dirname -- "$BASE_EXECUTABLE")" \
        'include-system-site-packages = true' "executable = $BASE_EXECUTABLE" > "$RUNTIME_PREFIX/pyvenv.cfg"
fi
mkdir -p "$RUNTIME_PREFIX/conda-meta" "$RUNTIME_PREFIX/etc/conda/activate.d"
touch "$RUNTIME_PREFIX/conda-meta/history"
cp "$ENV_PREFIX/etc/conda/activate.d/910c-cann.sh" "$RUNTIME_PREFIX/etc/conda/activate.d/910c-cann.sh"
"$RUNTIME_PREFIX/bin/python" -m pip install --no-deps --no-build-isolation --editable "$HERE/../prefix_tuning"
[[ -z $(find "$RUNTIME_PREFIX" -type l -print -quit) ]] || { echo 'Symlink found in vLLM runtime' >&2; exit 1; }
PIP_INDEX_URL="$PIP_INDEX_URL" bash "$HERE/tau2/setup.sh"
mkdir -p "$HERE/runtime"
"$ENV_PREFIX/bin/python" -m pip freeze > "$HERE/runtime/910c-bfcl-packages.txt"
PROJECT_PYTHON="$ENV_PREFIX/bin/python" bash -c 'source "$1/env.sh"; exec "$PROJECT_PYTHON" "$1/check_910c.py"' _ "$HERE"
