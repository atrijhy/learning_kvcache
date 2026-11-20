#!/usr/bin/env bash
# tau2-bench 环境 bootstrap（一次性）。
#
# tau2 要求 Python >=3.12,<3.14 且强依赖 litellm（会拖动 openai/httpx/tokenizers
# 等），无法复用主 .venv（Python 3.11 + vllm/transformers）。因此为 tau2 建一个
# 独立的 uv venv（Python 3.12），只装 tau2 及其依赖，与主环境完全隔离。
#
# 官方源码固定在 commit 363133a（v1.0.1）；判分逻辑原样使用，不做修改。
set -euo pipefail
export LITELLM_LOCAL_MODEL_COST_MAP=True

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-$SCRIPT_DIR/../runtime/tiktoken-cache}"
OFFICIAL_ROOT="${TAU2_OFFICIAL_ROOT:-$SCRIPT_DIR/official}"
if [[ $(uname -m) == aarch64 ]]; then
    DEFAULT_VENV="$SCRIPT_DIR/.venv-910c"
else
    DEFAULT_VENV="$SCRIPT_DIR/.venv"
fi
VENV_DIR="${TAU2_VENV:-$DEFAULT_VENV}"
TAU2_COMMIT="363133ada1936491fb5bcec33cd62c3518a99f65"
PY_VERSION="${TAU2_PYTHON_VERSION:-3.12}"
REPO_URL="https://github.com/sierra-research/tau2-bench"

if [[ $(uname -m) != aarch64 ]]; then
    command -v uv >/dev/null 2>&1 || { echo "uv 不在 PATH 中，请先安装 uv。" >&2; exit 1; }
fi

# 1) Pinned source; codeload archive also works when GitHub git transport is blocked.
SOURCE_MARKER="$OFFICIAL_ROOT/.source_commit"
if [[ -d "$OFFICIAL_ROOT/.git" ]]; then
    git -C "$OFFICIAL_ROOT" cat-file -e "$TAU2_COMMIT^{commit}" 2>/dev/null || git -C "$OFFICIAL_ROOT" fetch origin "$TAU2_COMMIT"
    git -C "$OFFICIAL_ROOT" checkout --quiet "$TAU2_COMMIT"
elif [[ -f "$SOURCE_MARKER" && "$(cat "$SOURCE_MARKER")" == "$TAU2_COMMIT" ]]; then
    test -f "$OFFICIAL_ROOT/src/tau2/cli.py"
else
    if [[ -d "$OFFICIAL_ROOT" && -n "$(ls -A "$OFFICIAL_ROOT")" ]]; then
        echo "Refusing to overwrite unverified tau2 source: $OFFICIAL_ROOT" >&2; exit 1
    fi
    mkdir -p "$OFFICIAL_ROOT"
    curl --fail --location --retry 3 "https://codeload.github.com/sierra-research/tau2-bench/tar.gz/$TAU2_COMMIT" |
        tar -xz --strip-components=1 -C "$OFFICIAL_ROOT"
fi
printf '%s\n' "$TAU2_COMMIT" > "$SOURCE_MARKER"

# 2) 专用 uv venv（Python 3.12），与主环境隔离。
if [[ $(uname -m) == aarch64 ]]; then
    # Keep the shared x86 venv intact. Copy the image's Python, with no symlinks
    # and no inherited packages, so LiteLLM cannot change the training stack.
    if [[ ! -x "$VENV_DIR/bin/python" ]]; then
        [[ ! -e "$VENV_DIR" ]] || { echo "Incomplete venv: $VENV_DIR" >&2; exit 1; }
        BASE_PYTHON=${BASE_PYTHON:-/usr/local/miniconda3/bin/python}
        BASE_EXECUTABLE=$(readlink -f -- "$BASE_PYTHON")
        PYTHON_ABI=$($BASE_PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        mkdir -p "$VENV_DIR/bin" "$VENV_DIR/lib/python$PYTHON_ABI/site-packages"
        install -m 0755 "$BASE_EXECUTABLE" "$VENV_DIR/bin/python"
        printf '%s\n' "home = $(dirname -- "$BASE_EXECUTABLE")" \
            'include-system-site-packages = false' "executable = $BASE_EXECUTABLE" > "$VENV_DIR/pyvenv.cfg"
        "$VENV_DIR/bin/python" -m ensurepip
    fi
elif [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "Creating tau2 venv (Python $PY_VERSION)..."
    uv venv --python "$PY_VERSION" "$VENV_DIR"
fi

# 3) editable 安装官方包（提供 tau2 命令 + 让 DATA_DIR 指向源码 data/）。
#    editable 安装后，tau2 从源码 checkout 解析 data/，无需再设 TAU2_DATA_DIR。
echo "Installing tau2 (editable) into $VENV_DIR ..."
if [[ $(uname -m) == aarch64 ]]; then
    "$VENV_DIR/bin/python" -m pip install \
        --index-url "${PIP_INDEX_URL:-https://mirrors.jd.com/pypi/web/simple}" -e "$OFFICIAL_ROOT"
    [[ -z $(find "$VENV_DIR" -type l -print -quit) ]] || { echo 'Symlink found in ARM venv' >&2; exit 1; }
else
    VIRTUAL_ENV="$VENV_DIR" uv pip install --python "$VENV_DIR/bin/python" -e "$OFFICIAL_ROOT"
fi

# 4) 校验：tau2 命令可用、版本、可列出 domain。
"$VENV_DIR/bin/tau2" --help >/dev/null 2>&1 || {
    echo "tau2 命令不可用，安装可能失败。" >&2
    exit 1
}
echo "Official commit: $TAU2_COMMIT"
echo "tau2 venv:       $VENV_DIR ($("$VENV_DIR/bin/python" --version))"
echo "tau2 ready."
