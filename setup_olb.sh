#!/usr/bin/env bash
# OpenLB Compilation and Environment Setup Script
# Author: Aditya001-max <kaushikadityaraj001@gmail.com>
# Repository: https://github.com/Aditya001-max/Computational-Fluid-Dynamics-Machine-Learning-Examples

set -euo pipefail

OLB_VERSION="olb-1.1r0"
OLB_ARCHIVE="${OLB_VERSION}.tgz"
OLB_URL="https://www.openlb.net/wp-content/uploads/2017/04/${OLB_ARCHIVE}"

echo "=== [1/3] Preparing OpenLB Environment (${OLB_VERSION}) ==="
if [ ! -d "${OLB_VERSION}" ]; then
    echo "Downloading ${OLB_ARCHIVE}..."
    if command -v curl >/dev/null 2>&1; then
        curl -fSL "${OLB_URL}" -o "${OLB_ARCHIVE}" || wget "${OLB_URL}" -O "${OLB_ARCHIVE}"
    else
        wget "${OLB_URL}" -O "${OLB_ARCHIVE}"
    fi
    echo "Extracting archive..."
    tar -xzf "${OLB_ARCHIVE}"
    rm -f "${OLB_ARCHIVE}"
fi

echo "=== [2/3] Compiling OpenLB Core Engine ==="
cd "${OLB_VERSION}"
make -j"$(nproc 2>/dev/null || echo 4)"
cd ..

echo "=== [3/3] Compiling 2D Cylinder Lattice Boltzmann Solver ==="
cd cylinder2d
make -j"$(nproc 2>/dev/null || echo 4)"
cd ..

echo "=== OpenLB Simulator Setup Completed Successfully! ==="
