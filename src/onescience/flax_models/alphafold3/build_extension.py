#!/usr/bin/env python3
"""
Build script for AlphaFold3 C++ extension and data files.
This script can be called during onescience installation to build the C++ extension and data files.
"""

import shutil
import subprocess
import sys
from pathlib import Path


def build_cpp_extension():
    """Build the AlphaFold3 C++ extension using CMake."""
    # Get the directory where this script is located (alphafold3 directory)
    alphafold3_dir = Path(__file__).parent.resolve()

    # Create build directory
    build_dir = alphafold3_dir / "build"
    build_dir.mkdir(exist_ok=True)

    try:
        # Configure CMake
        print(f"Configuring CMake in {build_dir}")
        subprocess.run(
            [
                "cmake",
                str(alphafold3_dir),
                "-DCMAKE_BUILD_TYPE=Release",
                "-DCMAKE_CXX_STANDARD=20",
                "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
            ],
            cwd=build_dir,
            check=True,
        )

        # Build
        print("Building C++ extension...")
        subprocess.run(
            ["cmake", "--build", ".", "--parallel"], cwd=build_dir, check=True
        )

        # Install to alphafold3 directory
        print(f"Installing C++ extension to {alphafold3_dir}")
        subprocess.run(
            ["cmake", "--install", ".", "--prefix", str(alphafold3_dir)],
            cwd=build_dir,
            check=True,
        )

        # Clean up temporary directories and files after successful installation
        print("Cleaning up temporary build files...")
        cleanup_dirs = [
            alphafold3_dir / "build",
            alphafold3_dir / "include",
            alphafold3_dir / "lib",
            alphafold3_dir / "lib64",
        ]

        for cleanup_dir in cleanup_dirs:
            if cleanup_dir.exists():
                print(f"  Removing {cleanup_dir}")
                shutil.rmtree(cleanup_dir, ignore_errors=True)

        print("✅ AlphaFold3 C++ extension built successfully!")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Error building AlphaFold3 C++ extension: {e}")
        return False
    except FileNotFoundError as e:
        print(f"❌ CMake not found. Please install CMake first: {e}")
        return False
    finally:
        # Always try to clean up build directory, even if build failed
        build_dir = alphafold3_dir / "build"
        if build_dir.exists():
            print("Cleaning up build directory...")
            shutil.rmtree(build_dir, ignore_errors=True)


def build_alphafold3_data():
    """Build AlphaFold3 data files (ccd.pickle and chemical_component_sets.pickle)."""
    try:
        print("Building AlphaFold3 data files...")
        # Import the build_data function
        from onescience.flax_models.alphafold3.build_data import build_data

        # Execute the build_data function
        build_data()

        print("✅ AlphaFold3 data files built successfully!")
        return True

    except ImportError as e:
        print(f"❌ Error importing build_data: {e}")
        return False
    except ValueError as e:
        print(f"❌ Error building data files: {e}")
        print(
            "This usually means libcifpp is not properly installed or components.cif file is missing."
        )
        print("Please install libcifpp first: https://github.com/PDB-REDO/libcifpp")
        return False
    except Exception as e:
        print(f"❌ Unexpected error building data files: {e}")
        return False


def build_all():
    """Build both C++ extension and data files."""
    print("🔨 Starting AlphaFold3 build process...")

    # Step 1: Build C++ extension
    cpp_success = build_cpp_extension()
    if not cpp_success:
        print("❌ Failed to build C++ extension. Aborting.")
        return False

    # Step 2: Build data files
    data_success = build_alphafold3_data()
    if not data_success:
        print("❌ Failed to build data files. C++ extension was built successfully.")
        return False

    print("🎉 All AlphaFold3 components built successfully!")
    return True


if __name__ == "__main__":
    success = build_all()
    sys.exit(0 if success else 1)
