#!/bin/bash
# Auto-build C++ MPC extension if not already installed
if ! python -c "import mpc_cpp_extension" 2>/dev/null; then
    echo "[Setup] Building C++ MPC extension..."
    pip install -e utama_core/motion_planning/src/mpc_cpp --no-build-isolation -q
    echo "[Setup] C++ MPC extension ready."
fi
