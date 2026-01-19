# Ensuring Your Conda Environment Works on the Server

## The Problem

When you run `pip freeze > requirements.txt` in a conda environment, it captures:
- ✅ Packages installed via pip (these work on the server)
- ❌ Packages installed via conda (may not be on PyPI)
- ❌ System packages (Ubuntu packages, not on PyPI)
- ❌ Packages with versions that don't exist on PyPI

## The Solution

Use the `generate_clean_requirements.py` script to create a server-compatible requirements.txt.

### Quick Start

```bash
# Activate your conda environment
conda activate rl-hockey

# Generate a clean requirements.txt
python resources/generate_clean_requirements.py --replace --test
```

This will:
1. Extract only pip-installed packages (not conda packages)
2. Validate each package exists on PyPI
3. Generate a clean `requirements.txt`
4. Test installation in a clean virtual environment
5. Replace your existing `requirements.txt` (with backup)

### Command Options

```bash
# Basic usage (creates requirements_clean.txt)
python resources/generate_clean_requirements.py

# Replace existing requirements.txt (creates backup)
python resources/generate_clean_requirements.py --replace

# Skip PyPI validation (faster, but less reliable)
python resources/generate_clean_requirements.py --no-validate

# Test installation in clean environment
python resources/generate_clean_requirements.py --test

# Full workflow (recommended)
python resources/generate_clean_requirements.py --replace --test
```

## Complete Workflow

### Step 1: Generate Clean Requirements

```bash
conda activate rl-hockey
cd /path/to/RL_CheungMaenzerAbraham_Hockey
python resources/generate_clean_requirements.py --replace --test
```

### Step 2: Review the Output

The script will show:
- ✅ Valid packages (written to requirements.txt)
- ❌ Excluded packages (system/conda packages)
- ❌ Failed packages (not on PyPI)

### Step 3: Handle Special Cases

If you need packages that aren't on PyPI:

1. **Conda-only packages**: Keep them in `environment.yml`, not `requirements.txt`
2. **System packages**: Add to container.def's apt-get install section
3. **Git packages**: Install separately in container.def (like hockey-env)

### Step 4: Rebuild Container

```bash
# On the server/cluster
bash resources/cluster_setup.sh
```

## What Gets Excluded

The script automatically excludes:
- System packages (python-apt, dbus-python, etc.)
- Packages with version 0.0.0
- Packages not found on PyPI

## Manual Validation

If you want to manually check a package:

```bash
# Check if package exists on PyPI
pip index versions package_name

# Try installing (dry run)
pip install package_name==version --dry-run
```

## Best Practices

1. **Use the script regularly**: Run it whenever you add new packages
2. **Test before deploying**: Always use `--test` flag
3. **Keep environment.yml separate**: Use it for conda-specific packages
4. **Document special cases**: Note any packages that need special handling

## Troubleshooting

### "Package not found on PyPI"

- Check if it's a conda package → move to environment.yml
- Check if it's a system package → add to container.def
- Check if version exists → update to available version

### "Installation test failed"

- Review error messages
- Check if dependencies are missing
- Verify package versions are correct

### "Package works locally but not on server"

- Likely a conda package → use the script to regenerate requirements.txt
- Check if it's in the exclusion list → verify it's actually needed

## Example Output

```
Generating clean requirements.txt from current environment...
================================================================================

1. Getting pip-installed packages...
   Found 150 packages installed via pip

2. Validating packages...
   ✓ numpy==1.26.4
   ✓ torch==2.7.1
   ✓ psutil==5.9.8
   ✗ defer==1.0.6 - Version 1.0.6 not found. Available: 1.0.4 (latest)
   ✗ python-apt==2.7.7 - System package (not on PyPI)

3. Writing clean requirements to requirements.txt...

================================================================================
SUMMARY
================================================================================
Total packages found: 150
Valid packages (written to requirements.txt): 145
Excluded packages (system/conda): 3
Failed validation (not on PyPI): 2

✓ Clean requirements.txt generated: requirements.txt
  This file contains 145 packages that are available on PyPI
```

## Integration with Container Build

The container setup already handles:
- Filtering out system packages
- Continuing build even if some packages fail
- Providing warnings about failed packages

Your clean requirements.txt will work seamlessly with the existing container setup.
