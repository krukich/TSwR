#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "============================================================"
echo "TSwR setup"
echo "Repo root: $(pwd)"
echo "============================================================"



if [ ! -d ".venv" ]; then
    echo "[setup] Creating .venv..."
    python3 -m venv .venv
fi


source .venv/bin/activate

echo "[setup] Python: $(which python)"
python --version

echo "[setup] Upgrading pip..."
python -m pip install --upgrade pip setuptools wheel


echo "[setup] Initializing/updating git submodules..."
git submodule update --init --recursive

if [ ! -d "rsl_rl" ]; then
    echo "[setup][ERROR] rsl_rl submodule directory not found."
    echo "Run:"
    echo "  git submodule add https://github.com/leggedrobotics/rsl_rl.git rsl_rl"
    exit 1
fi

if [ ! -d "genesis" ]; then
    echo "[setup][ERROR] genesis submodule directory not found."
    echo "Run:"
    echo "  git submodule add https://github.com/Genesis-Embodied-AI/Genesis.git genesis"
    exit 1
fi

echo "[setup] Checking rsl_rl version..."
git -C rsl_rl fetch --all --tags
git -C rsl_rl checkout 2ad79cf0caa85b91721abfe358105f869a784121

echo "[setup] Checking genesis version..."
git -C genesis fetch --all --tags
git -C genesis checkout 806d0a8


echo "[setup] Installing editable local packages..."
python -m pip install -e ./rsl_rl
python -m pip install -e ./genesis

if [ -f "requirements.txt" ]; then
    echo "[setup] Installing requirements.txt..."
    python -m pip install -r requirements.txt
else
    echo "[setup] No requirements.txt found, skipping."
fi

echo "[setup] Checking Go2 URDF assets..."

GO2_URDF="urdf/go2/urdf/go2.urdf"
GO2_DAE="urdf/go2/dae/base.dae"

if [ ! -f "$GO2_URDF" ] || [ ! -f "$GO2_DAE" ]; then
    echo "[setup] Go2 assets missing. Downloading go2_description..."

    mkdir -p third_party
    rm -rf third_party/go2_description

    git clone https://github.com/Unitree-Go2-Robot/go2_description.git third_party/go2_description

    mkdir -p urdf/go2/urdf
    mkdir -p urdf/go2/dae
    mkdir -p urdf/go2/meshes

    if [ -f "third_party/go2_description/urdf/go2_description.urdf" ]; then
        cp third_party/go2_description/urdf/go2_description.urdf "$GO2_URDF"
    elif [ -f "third_party/go2_description/urdf/go2.urdf" ]; then
        cp third_party/go2_description/urdf/go2.urdf "$GO2_URDF"
    else
        echo "[setup][ERROR] Could not find Go2 URDF in third_party/go2_description/urdf"
        find third_party/go2_description -name "*.urdf"
        exit 1
    fi

    if [ -d "third_party/go2_description/dae" ]; then
        cp -r third_party/go2_description/dae/* urdf/go2/dae/
    fi

    if [ -d "third_party/go2_description/meshes" ]; then
        cp -r third_party/go2_description/meshes/* urdf/go2/meshes/
    fi

    sed -i 's#package://go2_description/#../#g' "$GO2_URDF"
    sed -i 's#package://go2_description/dae/#../dae/#g' "$GO2_URDF"
    sed -i 's#package://go2_description/meshes/#../meshes/#g' "$GO2_URDF"
fi

if [ ! -f "$GO2_URDF" ]; then
    echo "[setup][ERROR] Missing $GO2_URDF"
    exit 1
fi

if [ ! -f "$GO2_DAE" ]; then
    echo "[setup][WARNING] Missing $GO2_DAE"
    echo "[setup][WARNING] If Genesis fails with Asset file not found: ../dae/base.dae,"
    echo "[setup][WARNING] check where base.dae was copied:"
    echo "  find urdf/go2 third_party/go2_description -name 'base.dae'"
fi


if [ ! -f "urdf/plane/plane.urdf" ]; then
    echo "[setup] No urdf/plane/plane.urdf found. This is OK if go2_env.py falls back to gs.morphs.Plane()."
fi


echo "============================================================"
echo "[setup] Import smoke test"
echo "============================================================"

python - <<'PY'
import genesis
import rsl_rl

print("genesis:", genesis.__file__)
print("rsl_rl:", rsl_rl.__file__)

try:
    import igl
    print("igl: ok")
except Exception as e:
    print("igl: not available:", repr(e))
PY

echo "============================================================"
echo "[setup] Done."
echo "Activate with:"
echo "  source .venv/bin/activate"
echo "Train with:"
echo "  python src/go2_train.py -e go2-walking -B 4096 --max_iterations 10000 --device cuda:0"
echo "============================================================"