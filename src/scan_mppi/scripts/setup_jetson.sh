#!/usr/bin/env bash
# Set up the car's Jetson Orin (JetPack 6) to run scan_mppi, on the GPU if possible.
#
#   bash setup_jetson.sh --check    report what this machine has and what --build needs;
#                                   installs nothing, no sudo. Run this first.
#   bash setup_jetson.sh            JAX from ~/jax_wheels/<ver> if it holds GPU wheels,
#                                   otherwise CPU JAX (works today; the planner drops to
#                                   fewer samples) plus instructions for the GPU build
#   bash setup_jetson.sh --build    build GPU jaxlib for the Orin from source (hours; once),
#                                   saving the wheels in ~/jax_wheels/<ver> for other cars
#   bash setup_jetson.sh --wheels DIR   install GPU wheels built earlier (jaxlib,
#                                   jax_cuda12_plugin, jax_cuda12_pjrt)
#   bash setup_jetson.sh --cpu      CPU JAX only
#   --ws DIR                        workspace to build scan_mppi in (default /home/f1tenth_ws)
#   --jobs N                        parallel compile jobs for --build (default 4; more can
#                                   run the Orin out of memory and freeze it)
#
# Why a source build: JAX publishes no Jetson wheels, and its generic aarch64 CUDA
# wheels install on an Orin but fail at the first kernel ("no kernel image is
# available"). Building for sm_87 against JetPack's CUDA 12.6 / cuDNN 9.3 with clang
# works from jax 0.5.3 (github.com/jax-ml/jax/issues/32166).
#
# Safe to re-run; every step checks first. Needs sudo only for apt (--build).
set -euo pipefail

JAX_VER=0.5.3        # first release known to build and run on Orin; same as the sim container
NUMPY_VER=1.26.4     # ROS 2 Humble's cv_bridge etc. break on numpy 2, which JAX would pull in
SCIPY_VER=1.11.4
ML_DTYPES_VER=0.5.1
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
WHEELS_DIR=$HOME/jax_wheels/$JAX_VER
ENV_FILE=$HOME/.scan_mppi_env
WS=/home/f1tenth_ws
MODE=auto
JOBS=4

while [[ $# -gt 0 ]]; do
    case $1 in
        --check) MODE=check ;;
        --build) MODE=build ;;
        --cpu) MODE=cpu ;;
        --wheels) MODE=wheels; WHEELS_DIR=$(realpath "$2"); shift ;;
        --ws) WS=$2; shift ;;
        --jobs) JOBS=$2; shift ;;
        -h|--help) sed -n '2,27p' "$0"; exit 0 ;;
        *) echo "unknown option $1 (see --help)" >&2; exit 1 ;;
    esac
    shift
done

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33mWARNING: %s\033[0m\n' "$*"; }
die()  { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }
pip_install() { python3 -m pip install --user --no-cache-dir "$@"; }
jax_on_gpu() {
    python3 - <<'PY' 2>/dev/null
import jax, jax.numpy as jnp
gpus = [d for d in jax.devices() if d.platform == 'gpu']
assert gpus, 'no GPU device'
x = jnp.ones((256, 256)); (x @ x).block_until_ready()   # runs a kernel: catches "no kernel image"
print(f'jax {jax.__version__} on {gpus[0].device_kind}')
PY
}

L4T=$(head -n1 /etc/nv_tegra_release 2>/dev/null | sed -n 's/^# R\([0-9]*\) (release), REVISION: \([0-9.]*\).*/\1.\2/p' || true)
PY=$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null || true)
CUDA_VER=$(python3 -c "import json; print(json.load(open('$CUDA_HOME/version.json'))['cuda']['version'])" 2>/dev/null \
           || "$CUDA_HOME/bin/nvcc" --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p' || true)
CUDNN_H=$(ls /usr/include/cudnn_version*.h /usr/include/aarch64-linux-gnu/cudnn_version*.h 2>/dev/null | head -n1 || true)
CUDNN_VER=$([[ -n $CUDNN_H ]] && awk '/#define CUDNN_(MAJOR|MINOR|PATCHLEVEL) / {v = v (v ? "." : "") $3} END {print v}' "$CUDNN_H" || true)
version_ge() { [[ -n $2 && $(printf '%s\n' "$1" "$2" | sort -V | head -n1) == "$1" ]]; }   # $2 >= $1

# ------------------------------------------------------------------ --check
if [[ $MODE == check ]]; then
    fails=0
    ok()   { printf '  \033[32mok\033[0m    %s\n' "$*"; }
    bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; fails=$((fails + 1)); }
    note() { printf '  \033[33mnote\033[0m  %s\n' "$*"; }
    find_lib() {
        ldconfig -p 2>/dev/null | grep -q "$1" && return 0
        local d; for d in "$CUDA_HOME"/lib64 "$CUDA_HOME"/extras/CUPTI/lib64 /usr/lib/aarch64-linux-gnu; do
            compgen -G "$d/$1*" >/dev/null && return 0
        done; return 1
    }

    say "Platform"
    [[ $(uname -m) == aarch64 ]] && ok "aarch64" || bad "$(uname -m): not the Jetson"
    model=$({ tr -d '\0' < /proc/device-tree/model; } 2>/dev/null || true)
    [[ $model == *Orin* ]] && ok "$model" || note "board '${model:-unknown}' (expected an Orin, GPU sm_87)"
    [[ ${L4T%%.*} -ge 36 ]] 2>/dev/null && ok "L4T $L4T (JetPack 6)" || bad "L4T '${L4T:-none}': need JetPack 6 (R36)"
    [[ $PY == 3.10 ]] && ok "python $PY" || bad "python '${PY:-none}': need 3.10"
    python3 -m pip --version >/dev/null 2>&1 && ok "pip" || bad "pip missing: sudo apt install python3-pip"
    [[ -f /opt/ros/humble/setup.bash ]] && ok "ROS 2 Humble" || bad "/opt/ros/humble not found"
    [[ -d $WS/src ]] && ok "workspace $WS" || note "workspace $WS not found (--ws DIR)"
    [[ -f $WS/src/pure_pursuit/waypoints/aims_closed.csv ]] && ok "aims_closed.csv" \
        || note "$WS/src/pure_pursuit/waypoints/aims_closed.csv missing (copy it, or set waypoints_file in car.yaml)"

    say "CUDA (JetPack), needed by GPU JAX at run time"
    version_ge 12.6 "$CUDA_VER" && ok "CUDA $CUDA_VER" || bad "CUDA '${CUDA_VER:-none}': need >= 12.6 (JetPack 6.1+)"
    [[ ${CUDNN_VER%%.*} == 9 ]] && ok "cuDNN $CUDNN_VER" || bad "cuDNN '${CUDNN_VER:-none}': need 9 (JetPack 6.1+)"
    for lib in libcudart.so.12 libcublas.so.12 libcublasLt.so.12 libcufft.so.11 libcusolver.so.11 \
               libcusparse.so.12 libnvJitLink.so.12 libcudnn.so.9; do
        find_lib $lib && ok "$lib" || bad "$lib not found (in $CUDA_HOME/lib64 or the linker cache)"
    done
    find_lib libcupti.so.12 && ok "libcupti.so.12" || note "libcupti.so.12 not found (profiling only)"
    # XLA compiles its kernels with ptxas at run time; without it JAX fails on the first jit
    if "$CUDA_HOME/bin/ptxas" --help 2>/dev/null | grep -q sm_87 || "$CUDA_HOME/bin/ptxas" --list-gpu-arch 2>/dev/null | grep -q sm_87; then
        ok "ptxas (sm_87)"
    else bad "$CUDA_HOME/bin/ptxas missing or without sm_87 (sudo apt install cuda-nvcc-12-6)"; fi
    if [[ -x $CUDA_HOME/bin/nvcc ]]; then
        tmp=$(mktemp -d)
        cat > "$tmp/k.cu" <<'CU'
#include <cstdio>
__global__ void k(int *x) { x[threadIdx.x] = threadIdx.x; }
int main() {
    int *d, h[32]; cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) { printf("no CUDA device\n"); return 1; }
    cudaMalloc(&d, sizeof h); k<<<1, 32>>>(d);
    cudaError_t e = cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
    printf("%s sm_%d%d: %s\n", p.name, p.major, p.minor, e == cudaSuccess && h[31] == 31 ? "kernel ran" : cudaGetErrorString(e));
    return e == cudaSuccess && h[31] == 31 ? 0 : 1;
}
CU
        if out=$("$CUDA_HOME/bin/nvcc" -arch=sm_87 -o "$tmp/k" "$tmp/k.cu" 2>&1 && "$tmp/k" 2>&1); then ok "GPU: $out"
        else bad "test CUDA kernel: $(echo "$out" | tail -n1)"; fi
        rm -rf "$tmp"
    else bad "$CUDA_HOME/bin/nvcc missing (sudo apt install cuda-nvcc-12-6); --build also needs it"; fi

    say "Python packages now installed"
    python3 - <<'PY' || true
import importlib
for m in ('numpy', 'scipy', 'ml_dtypes', 'yaml', 'jax', 'jaxlib', 'jax_cuda12_plugin'):
    try: print(f'  {m:18s} {getattr(importlib.import_module(m), "__version__", "installed")}')
    except Exception as e: print(f'  {m:18s} - ({type(e).__name__})')
PY
    if python3 -c 'import numpy, sys; sys.exit(numpy.__version__.startswith("2."))' 2>/dev/null; then :
    else note "numpy 2 (or none) installed; setup pins $NUMPY_VER, which ROS Humble needs"; fi
    if jax_on_gpu >/dev/null 2>&1; then ok "JAX already runs on the GPU: no build needed"; fi

    say "Only for --build"
    mem_gb=$(awk '/MemTotal|SwapTotal/ {s += $2} END {printf "%d", s / 1048576}' /proc/meminfo)
    disk_gb=$(df -BG --output=avail "$HOME" | tail -n1 | tr -dc 0-9)
    [[ $disk_gb -ge 25 ]] && ok "free disk in $HOME: ${disk_gb} GB" || bad "free disk in $HOME: ${disk_gb} GB, need ~25"
    [[ $mem_gb -ge 16 ]] && ok "RAM + swap: ${mem_gb} GB" || bad "RAM + swap ${mem_gb} GB, want >= 16: add swap
          sudo fallocate -l 16G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile"
    for c in git gcc g++; do command -v $c >/dev/null && ok "$c" || note "$c missing (--build installs it)"; done
    dpkg -s libstdc++-12-dev >/dev/null 2>&1 && ok "libstdc++-12-dev" || note "libstdc++-12-dev missing (--build installs it)"
    if command -v clang-18 >/dev/null; then ok "clang-18"
    else note "clang-18 missing (--build installs it from apt.llvm.org; Ubuntu's clang 14 is too old)"; fi
    sudo -n true 2>/dev/null && ok "sudo without a password" || note "--build will ask for your sudo password (apt)"
    for url in https://github.com https://apt.llvm.org https://pypi.org https://storage.googleapis.com; do
        curl -sI --max-time 8 "$url" >/dev/null 2>&1 && ok "reach $url" || bad "cannot reach $url (the build downloads from it)"
    done
    command -v nvpmodel >/dev/null && note "power mode: $(nvpmodel -q 2>/dev/null | head -n1) (build and run fastest with: sudo nvpmodel -m 0 && sudo jetson_clocks)"

    echo
    if [[ $fails == 0 ]]; then echo "All checks passed: ready for  bash $0 --build"
    else echo "$fails check(s) failed; fix those before --build (the default mode still installs CPU JAX)"; fi
    exit $(( fails > 0 ))
fi

# ---------------------------------------------------------------- platform
say "1/6  Platform"
[[ $(uname -m) == aarch64 ]] || die "not an aarch64 machine; this script is for the car's Jetson"
[[ -n $L4T ]] || die "/etc/nv_tegra_release not found: not a Jetson?"
echo "L4T $L4T   python $PY   CUDA ${CUDA_VER:-none}   cuDNN ${CUDNN_VER:-none}"
[[ ${L4T%%.*} -ge 36 ]] || die "L4T $L4T is JetPack 5 (Python 3.8); JAX $JAX_VER needs JetPack 6 (Python 3.10)"
[[ $PY == 3.10 ]] || die "expected Python 3.10 on JetPack 6, found $PY"

gpu_ok=1
if ! version_ge 12.6 "$CUDA_VER"; then
    warn "GPU JAX needs CUDA >= 12.6 (JetPack 6.1+); found '${CUDA_VER:-none}'"; gpu_ok=0
fi
if [[ ${CUDNN_VER%%.*} != 9 ]]; then
    warn "GPU JAX needs cuDNN 9 (JetPack 6.1+); found '${CUDNN_VER:-none}'"; gpu_ok=0
fi
if [[ $gpu_ok == 0 && $MODE != cpu ]]; then
    [[ $MODE == auto ]] || die "cannot do --$MODE without CUDA 12.6 + cuDNN 9; upgrade JetPack or use --cpu"
    warn "falling back to CPU JAX"; MODE=cpu
fi

# ------------------------------------------------------- python packages
say "2/6  Python packages"
python3 -m pip --version >/dev/null 2>&1 || die "pip missing: sudo apt install python3-pip"
pip_install "numpy==$NUMPY_VER" "scipy==$SCIPY_VER" "ml_dtypes==$ML_DTYPES_VER" pyyaml

# ------------------------------------------------------------- environment
say "3/6  Environment ($ENV_FILE)"
# JetPack's CUDA libraries, and no up-front grab of 75% of memory: on the Orin
# the GPU shares RAM with ROS. (scan_mppi's launch file sets both for the node too.)
cat > "$ENV_FILE" <<ENV
export LD_LIBRARY_PATH=$CUDA_HOME/lib64\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV
grep -qF "$ENV_FILE" "$HOME/.bashrc" || echo "[ -f $ENV_FILE ] && source $ENV_FILE" >> "$HOME/.bashrc"
# shellcheck disable=SC1090
source "$ENV_FILE"

# ------------------------------------------------------------------- JAX

have_wheels() { ls "$1"/jaxlib-"$JAX_VER"-*.whl "$1"/jax_cuda12_plugin-"$JAX_VER"-*.whl "$1"/jax_cuda12_pjrt-"$JAX_VER"-*.whl >/dev/null 2>&1; }

install_gpu_wheels() {
    have_wheels "$1" || die "no jaxlib / jax_cuda12_plugin / jax_cuda12_pjrt $JAX_VER wheels in $1"
    # The plugin without its [with-cuda] extra: NVIDIA's pip CUDA libraries are for
    # servers and crash on the Orin; JetPack's own (LD_LIBRARY_PATH above) are used.
    pip_install --force-reinstall --no-deps "$1"/jaxlib-"$JAX_VER"-*.whl \
        "$1"/jax_cuda12_plugin-"$JAX_VER"-*.whl "$1"/jax_cuda12_pjrt-"$JAX_VER"-*.whl
    pip_install "jax==$JAX_VER" "numpy==$NUMPY_VER" "scipy==$SCIPY_VER" "ml_dtypes==$ML_DTYPES_VER" opt_einsum
}

build_gpu_wheels() {
    local src=$HOME/jax_build/jax
    local mem_gb; mem_gb=$(awk '/MemTotal|SwapTotal/ {s += $2} END {printf "%d", s / 1048576}' /proc/meminfo)
    local disk_gb; disk_gb=$(df -BG --output=avail "$HOME" | tail -n1 | tr -dc 0-9)
    echo "RAM + swap ${mem_gb} GB, free disk ${disk_gb} GB, $JOBS jobs"
    [[ $disk_gb -ge 25 ]] || die "need ~25 GB free in $HOME for the build"
    [[ $mem_gb -ge 16 ]] || warn "under 16 GB RAM + swap: add swap or the build may freeze, e.g.
      sudo fallocate -l 16G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile"
    say "Build tools (sudo apt)"
    sudo apt-get update
    sudo apt-get install -y git wget gnupg lsb-release software-properties-common build-essential \
        python3-dev libstdc++-12-dev
    # clang 17+ is needed (clang 14 from Ubuntu 22.04 fails on an SVE intrinsic)
    if ! command -v clang-18 >/dev/null; then
        wget -qO /tmp/llvm.sh https://apt.llvm.org/llvm.sh
        sudo bash /tmp/llvm.sh 18
    fi
    [[ -d $src ]] || git clone --depth 1 --branch "jax-v$JAX_VER" https://github.com/jax-ml/jax "$src"
    say "Building jaxlib $JAX_VER for sm_87 (this takes hours)"
    (cd "$src" && python3 build/build.py build \
        --wheels=jaxlib,jax-cuda-plugin,jax-cuda-pjrt \
        --cuda_version=12.6.0 --cudnn_version=9.3.0 --cuda_compute_capabilities=sm_87 \
        --build_cuda_with_clang --clang_path="$(command -v clang-18)" \
        --bazel_options=--copt=-Wno-c23-extensions --bazel_options=--jobs="$JOBS" \
        --output_path="$WHEELS_DIR")
}

say "4/6  JAX ($MODE)"
if [[ $MODE != cpu ]] && jax_on_gpu; then
    echo "JAX already runs on the GPU; nothing to do"
else
    case $MODE in
        auto)
            if have_wheels "$WHEELS_DIR"; then install_gpu_wheels "$WHEELS_DIR"
            else MODE=cpu; pip_install "jax==$JAX_VER" "jaxlib==$JAX_VER" "numpy==$NUMPY_VER" "ml_dtypes==$ML_DTYPES_VER"
            fi ;;
        wheels) install_gpu_wheels "$WHEELS_DIR" ;;
        build) have_wheels "$WHEELS_DIR" || build_gpu_wheels; install_gpu_wheels "$WHEELS_DIR" ;;
        cpu) pip_install "jax==$JAX_VER" "jaxlib==$JAX_VER" "numpy==$NUMPY_VER" "ml_dtypes==$ML_DTYPES_VER" ;;
    esac
fi

# ------------------------------------------------------------------ check
say "5/6  Check"
python3 -c "import numpy, scipy; assert numpy.__version__.startswith('1.'), numpy.__version__"
if jax_on_gpu; then
    ON=gpu
else
    ON=cpu
    python3 -c "import jax; print('jax', jax.__version__, 'on', jax.devices()[0].platform)"
    [[ $MODE == cpu ]] || die "JAX does not run on the GPU (see the output above)"
fi
PKG=$(cd "$(dirname "$0")/.." && pwd)
if [[ -f $WS/src/pure_pursuit/waypoints/aims_closed.csv ]]; then
    echo "scan_mppi solve time on this machine:"
    PYTHONPATH=$PKG python3 -c "
from scan_mppi.controller import Controller
Controller('$WS/src/pure_pursuit/waypoints/aims_closed.csv', False, 4.0, 0.6).warm_up()"
else
    warn "$WS/src/pure_pursuit/waypoints/aims_closed.csv not found; skipping the solve-time check"
fi

# ------------------------------------------------------------------ build
say "6/6  ROS package"
if [[ -d $WS/src ]]; then
    [[ -e $WS/src/scan_mppi ]] || warn "scan_mppi is not in $WS/src; copy it there first"
    if [[ -e $WS/src/scan_mppi ]]; then
        # shellcheck disable=SC1091
        set +u; source /opt/ros/humble/setup.bash; set -u
        (cd "$WS" && colcon build --packages-select scan_mppi)
    fi
else
    warn "workspace $WS not found (use --ws)"
fi

echo
echo "Done: JAX on the $ON."
[[ $ON == gpu ]] || echo "For the GPU (once, takes hours): bash $0 --build    then copy $WHEELS_DIR to other cars and use --wheels."
echo "For the fastest solves: sudo nvpmodel -m 0 && sudo jetson_clocks"
echo "Run: source $WS/install/setup.bash && ros2 launch scan_mppi scan_mppi.launch.py config:=car"
