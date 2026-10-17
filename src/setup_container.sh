#!/usr/bin/env bash
# Set up a fresh f1tenth_gym_ros container so the opp_aware_mppi MPPI node can run.
#
# Run INSIDE the container after rocker starts it:
#     bash /sim_ws/src/setup_container.sh
#
# Safe to re-run: every step is skipped if it is already satisfied.
set -euo pipefail

JAX_VER=0.5.3      # last JAX release supporting Python 3.10
NUMPY_VER=1.26.4   # JAX's wheels pull numpy 2.x, which breaks f110-gym
SCIPY_VER=1.11.4
ML_DTYPES_VER=0.5.1

WS=/sim_ws
PKGS=(f1tenth_gym_ros raceline_msgs mppi_example mppi_bringup)
DO_BUILD=1
[[ "${1:-}" == "--no-build" ]] && DO_BUILD=0

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

if [[ ! -d $WS/src ]]; then
    echo "ERROR: $WS/src not found - run this inside the sim container." >&2
    exit 1
fi

say "1/5  DNS"
# The stock image ships a resolv.conf pointing at an unreachable resolver, so
# pip cannot reach pypi until this is patched.
if python3 -c 'import socket; socket.gethostbyname("pypi.org")' 2>/dev/null; then
    echo "    pypi.org resolves, leaving resolv.conf alone"
else
    echo "nameserver 8.8.8.8" >/etc/resolv.conf
    echo "    patched /etc/resolv.conf -> 8.8.8.8"
    python3 -c 'import socket; socket.gethostbyname("pypi.org")' >/dev/null
fi

say "2/5  JAX $JAX_VER"
if python3 -c "import jax, sys; sys.exit(0 if jax.__version__ == '$JAX_VER' else 1)" 2>/dev/null; then
    echo "    already installed"
else
    pip3 install --no-cache-dir "jax[cuda12]==$JAX_VER"
fi

say "3/5  CUDA runtime wheels (~3 GB on first run)"
# Must be a SEPARATE pip call from the jax[cuda12] install above. Asking for
# both at once makes pip's resolver see jax-cuda12-plugin[with-cuda,with-cuda]
# and abort with "AssertionError: Candidate is not for this requirement".
if python3 -c 'import nvidia.cudnn, nvidia.cublas' 2>/dev/null; then
    echo "    already installed"
else
    pip3 install --no-cache-dir "jax-cuda12-plugin[with-cuda]==$JAX_VER"
fi

say "4/5  Pin numpy / scipy / ml_dtypes back"
pip3 install --no-cache-dir \
    "numpy==$NUMPY_VER" "scipy==$SCIPY_VER" "ml_dtypes==$ML_DTYPES_VER"

say "5/5  Verify"
python3 - <<'PY'
import jax, jax.numpy as jnp, numpy, scipy
print(f"    numpy {numpy.__version__}  scipy {scipy.__version__}  jax {jax.__version__}")
devs = jax.devices()
print(f"    jax devices: {devs}")
float((jnp.ones((512, 512)) @ jnp.ones((512, 512))).sum())
import f110_gym  # noqa: F401
print("    f110_gym imports OK")
if not any(d.platform == "gpu" for d in devs):
    raise SystemExit("    WARNING: JAX fell back to CPU - MPPI will be slow")
PY

if (( DO_BUILD )); then
    say "Building workspace"
    set +u  # ROS setup.bash reads unbound vars such as AMENT_TRACE_SETUP_FILES
    # shellcheck disable=SC1091
    source /opt/ros/humble/setup.bash
    set -u
    cd "$WS"
    colcon build --packages-select "${PKGS[@]}"
fi

cat <<'EOF'

Done. In two separate terminals:

    docker exec -it f1tenth bash -lc 'source /opt/ros/humble/setup.bash && source /sim_ws/install/setup.bash && ros2 launch f1tenth_gym_ros gym_bridge_launch.py'
    docker exec -it f1tenth bash -lc 'source /opt/ros/humble/setup.bash && source /sim_ws/install/setup.bash && ros2 launch mppi_bringup sim_lev_blocked.launch.py'

NOTE: rocker starts the container with --rm, so exiting the interactive shell
deletes it along with everything installed above. Leave it open and use
`docker exec` from other terminals.
EOF
