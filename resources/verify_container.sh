#!/bin/bash
# Script to verify what packages are installed in the container
# Run this on the cluster after building the container

PROJECT_DIR="$HOME/RL_CheungMaenzerAbraham_Hockey"
CONTAINER="$PROJECT_DIR/singularity_build/rl_hockey.simg"

if [ ! -f "$CONTAINER" ]; then
    echo "Error: Container not found at $CONTAINER"
    exit 1
fi

echo "Checking installed packages in container..."
echo ""

# First, verify we're using the container's venv Python
echo "=== Environment Check ==="
singularity exec "$CONTAINER" /bin/bash -c "source /venv/bin/activate && which python3 && python3 --version"
echo ""

# Check if requirements.txt was copied into container
echo "=== Checking if requirements.txt exists in container ==="
if singularity exec "$CONTAINER" test -f /tmp/requirements.txt; then
    echo "✓ requirements.txt found in container at /tmp/requirements.txt"
    echo "First 10 lines of requirements.txt in container:"
    singularity exec "$CONTAINER" head -10 /tmp/requirements.txt
    echo ""
    echo "Total lines: $(singularity exec "$CONTAINER" wc -l < /tmp/requirements.txt)"
else
    echo "✗ requirements.txt NOT found in container!"
fi
echo ""

# List some packages that should be in requirements.txt
echo "=== Checking for key packages from requirements.txt ==="
singularity exec "$CONTAINER" /bin/bash -c "source /venv/bin/activate && python3 << 'PYEOF'
import sys
print(f'Python path: {sys.executable}')
print(f'Python prefix: {sys.prefix}')
print()
packages = ['pandas', 'scipy', 'moviepy', 'cattrs', 'imageio', 'requests_cache', 'torch', 'numpy', 'gymnasium']
for pkg in packages:
    try:
        mod = __import__(pkg)
        version = getattr(mod, '__version__', 'unknown')
        print(f'✓ {pkg} is installed (version: {version})')
    except ImportError as e:
        print(f'✗ {pkg} is NOT installed')
PYEOF"

echo ""
echo "=== Checking pip list (first 40 packages) ==="
singularity exec "$CONTAINER" /bin/bash -c "source /venv/bin/activate && pip list | head -40"
