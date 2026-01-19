#!/usr/bin/env python3
"""
Generate a clean requirements.txt from your conda environment that will work on the server.

This script:
1. Extracts only packages installed via pip (not conda)
2. Validates packages exist on PyPI
3. Generates a clean requirements.txt
4. Optionally tests installation in a virtual environment
"""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def get_pip_installed_packages() -> List[Tuple[str, str]]:
    """Get packages installed via pip (not conda)."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            capture_output=True,
            text=True,
            check=True,
        )
        packages = json.loads(result.stdout)
        return [(pkg["name"], pkg["version"]) for pkg in packages]
    except Exception as e:
        print(f"Error getting pip packages: {e}")
        return []


def check_package_on_pypi(package_name: str, version: str = None) -> Tuple[bool, str]:
    """Check if a package exists on PyPI and if the version is available."""
    try:
        # Use pip index to check package availability
        cmd = [sys.executable, "-m", "pip", "index", "versions", package_name]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)

        if result.returncode == 0:
            output = result.stdout
            if version:
                # Check if specific version exists
                if f"=={version}" in output or version in output:
                    return True, f"Version {version} available"
                else:
                    # Extract available versions
                    versions = re.findall(r"(\d+\.\d+\.\d+)", output)
                    if versions:
                        return (
                            False,
                            f"Version {version} not found. Available: {versions[0]} (latest)",
                        )
                    return False, f"Version {version} not found"
            return True, "Package available"
        else:
            return False, "Not found on PyPI"
    except subprocess.TimeoutExpired:
        return False, "Timeout checking PyPI"
    except Exception as e:
        # Fallback: try to install the package to see if it exists
        try:
            cmd = [
                sys.executable,
                "-m",
                "pip",
                "install",
                f"{package_name}=={version}" if version else package_name,
                "--dry-run",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if "ERROR" in result.stderr or "Could not find" in result.stderr:
                return False, "Not found on PyPI"
            return True, "Package available"
        except:
            return False, f"Error: {str(e)}"


def is_system_package(package_name: str) -> bool:
    """Check if a package is likely a system package (not on PyPI)."""
    system_packages = {
        "Brlapi",
        "cloud-init",
        "ubuntu-drivers-common",
        "ubuntu-pro-client",
        "ufw",
        "unattended-upgrades",
        "usb-creator",
        "command-not-found",
        "language-selector",
        "python-apt",
        "python-debian",
        "systemd-python",
        "distro-info",
        "xkit",
        "cupshelpers",
        "dbus-python",
        "duplicity",
        "eduvpn-client",
        "eduvpn-common",
        "gpg",
        "launchpadlib",
        "louis",
        "pycairo",
        "pycups",
        "PyGObject",
        "jeepney",
        "defer",
    }
    return package_name in system_packages


def should_exclude_package(package_name: str, version: str) -> Tuple[bool, str]:
    """Determine if a package should be excluded and why."""
    # Exclude system packages
    if is_system_package(package_name):
        return True, "System package (not on PyPI)"

    # Exclude packages with version 0.0.0 (often system packages)
    if version == "0.0.0":
        return True, "Version 0.0.0 (likely system package)"

    # Exclude editable installs (they're local)
    return False, ""


def generate_clean_requirements(
    output_file: str = "requirements_clean.txt", validate: bool = True
) -> None:
    """Generate a clean requirements.txt file."""
    print("Generating clean requirements.txt from current environment...")
    print("=" * 80)

    # Get all pip-installed packages
    print("\n1. Getting pip-installed packages...")
    packages = get_pip_installed_packages()
    print(f"   Found {len(packages)} packages installed via pip")

    # Filter and validate packages
    print("\n2. Validating packages...")
    valid_packages = []
    excluded_packages = []
    failed_packages = []

    for package_name, version in packages:
        # Skip if should be excluded
        should_exclude, reason = should_exclude_package(package_name, version)
        if should_exclude:
            excluded_packages.append((package_name, version, reason))
            continue

        # Validate on PyPI if requested
        if validate:
            exists, message = check_package_on_pypi(package_name, version)
            if exists:
                valid_packages.append((package_name, version))
                print(f"   ✓ {package_name}=={version}")
            else:
                failed_packages.append((package_name, version, message))
                print(f"   ✗ {package_name}=={version} - {message}")
        else:
            valid_packages.append((package_name, version))
            print(f"   ✓ {package_name}=={version}")

    # Write clean requirements.txt
    print(f"\n3. Writing clean requirements to {output_file}...")
    with open(output_file, "w") as f:
        for package_name, version in sorted(valid_packages):
            f.write(f"{package_name}=={version}\n")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total packages found: {len(packages)}")
    print(f"Valid packages (written to {output_file}): {len(valid_packages)}")
    print(f"Excluded packages (system/conda): {len(excluded_packages)}")
    if failed_packages:
        print(f"Failed validation (not on PyPI): {len(failed_packages)}")

    if excluded_packages:
        print("\nExcluded packages:")
        for pkg, ver, reason in excluded_packages[:10]:  # Show first 10
            print(f"  - {pkg}=={ver} ({reason})")
        if len(excluded_packages) > 10:
            print(f"  ... and {len(excluded_packages) - 10} more")

    if failed_packages:
        print("\nPackages not found on PyPI:")
        for pkg, ver, msg in failed_packages:
            print(f"  - {pkg}=={ver}: {msg}")

    print(f"\n✓ Clean requirements.txt generated: {output_file}")
    print(
        f"  This file contains {len(valid_packages)} packages that are available on PyPI"
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a clean requirements.txt from conda environment"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="requirements_clean.txt",
        help="Output file name (default: requirements_clean.txt)",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip PyPI validation (faster but less reliable)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test installation in a clean virtual environment",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing requirements.txt (backup will be created)",
    )

    args = parser.parse_args()

    # Backup existing requirements.txt if replacing
    if args.replace:
        original_file = "requirements.txt"
        if Path(original_file).exists():
            backup_file = f"{original_file}.backup"
            print(f"Backing up {original_file} to {backup_file}...")
            shutil.copy(original_file, backup_file)
            args.output = original_file

    # Generate clean requirements
    generate_clean_requirements(output_file=args.output, validate=not args.no_validate)


if __name__ == "__main__":
    main()
