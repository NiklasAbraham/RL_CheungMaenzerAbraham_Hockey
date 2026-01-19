# Understanding requirements.txt Issues

## Why Packages "Don't Exist" on PyPI

When you run `pip freeze > requirements.txt` in a conda environment, it captures **everything** installed in that environment, including:

1. **Conda-installed packages**: Packages installed via `conda install` that may not be on PyPI or have different versions
2. **System packages**: Ubuntu/Debian packages installed via `apt` (like `python-apt`, `dbus-python`, etc.)
3. **Local packages**: Packages installed from local sources
4. **Incorrect versions**: Versions that were available when installed but no longer exist on PyPI

### Example: `defer==1.0.6`

- You have `defer==1.0.6` in your conda environment
- PyPI only has versions up to `1.0.4`
- This likely means:
  - It was installed via conda (conda-forge may have a newer version)
  - Or it's a system package
  - Or the version was available when you installed it but has been removed

## How to Fix This

### Option 1: Use the Container's Exclusion List (Current Approach)

The container setup already handles this by:
- Filtering out known system packages
- Continuing installation even if some packages fail
- Providing warnings about failed packages

**This is the recommended approach** - the container will work even with some packages missing.

### Option 2: Clean Your requirements.txt

If you want a cleaner requirements.txt that only includes PyPI packages:

1. **Identify actual dependencies**: Look at your imports in the codebase
2. **Test in a clean environment**: Create a fresh venv and install only what you need
3. **Use pip-tools**: Generate a requirements.txt from a requirements.in file

### Option 3: Use Both Conda and Pip

For packages that are conda-only:
- Keep them in `environment.yml` (conda)
- Only put PyPI packages in `requirements.txt` (pip)

## Common Problematic Packages

These packages are often in `pip freeze` but aren't on PyPI:

- `defer` - May be conda-only or system package
- `Brlapi`, `python-apt`, `dbus-python` - Ubuntu system packages
- `cloud-init`, `ubuntu-drivers-common` - System packages
- `cupshelpers`, `PyGObject`, `pycairo` - System GUI packages
- Packages with `==0.0.0` - Often system packages

## Current Container Setup

The container already excludes these problematic packages. If you encounter a new one:

1. Add it to the exclusion list in `container.def` (the `EXCLUDE_PATTERNS` section)
2. Or install it via `apt-get` if it's a system package
3. Or remove it from requirements.txt if it's not needed

## Checking if a Package is Needed

To check if a package is actually used in your code:

```bash
# Search for imports
grep -r "import defer" src/
grep -r "from defer" src/

# If no results, the package isn't used and can be excluded
```

## Best Practice

Instead of `pip freeze`, maintain a `requirements.in` file with only the packages you directly depend on, then use `pip-compile` to generate `requirements.txt` with all dependencies pinned.
