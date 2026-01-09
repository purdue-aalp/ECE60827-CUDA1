#!/bin/bash

#############################################################################
# ECE 60827 CUDA Programming Lab 1 - Automated Grading Script
#
# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                          ⚠️  DO NOT EDIT  ⚠️                          ║
# ║                                                                       ║
# ║  This file is part of the automated grading system.                  ║
# ║  Any modifications to this file may result in grading failures.      ║
# ║                                                                       ║
# ║  Students: Please implement your solutions in:                       ║
# ║    - src/cudaLib.cu   (GPU implementations)                          ║
# ║    - src/cpuLib.cpp   (CPU implementations)                          ║
# ║                                                                       ║
# ║  Usage: ./autograder/grade.sh                                        ║
# ╚═══════════════════════════════════════════════════════════════════════╝
#
#############################################################################

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Determine the project root directory (parent of autograder folder)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"
GRADER_EXECUTABLE="${BUILD_DIR}/grader"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     ECE 60827 CUDA Lab 1 - Automated Grading Script       ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

#############################################################################
# Step 1: Check for CUDA environment
#############################################################################
echo -e "${YELLOW}[Step 1/4]${NC} Checking CUDA environment..."

if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}✗ ERROR: nvcc not found in PATH${NC}"
    echo "Please load CUDA module: module load cuda"
    exit 1
fi

NVCC_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9.]*\).*/\1/p')
echo -e "${GREEN}✓${NC} CUDA toolkit found (version: ${NVCC_VERSION})"

# Check for GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}⚠ WARNING: nvidia-smi not found - GPU may not be available${NC}"
else
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    if [ "$GPU_COUNT" -eq 0 ]; then
        echo -e "${RED}✗ ERROR: No CUDA-capable GPU detected${NC}"
        exit 1
    fi
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    echo -e "${GREEN}✓${NC} GPU detected: ${GPU_NAME}"
fi

#############################################################################
# Step 2: Clean previous build
#############################################################################
echo ""
echo -e "${YELLOW}[Step 2/4]${NC} Cleaning previous build..."

if [ -d "${BUILD_DIR}" ]; then
    echo "Removing existing build directory..."
    rm -rf "${BUILD_DIR}"
fi

mkdir -p "${BUILD_DIR}"
echo -e "${GREEN}✓${NC} Build directory created"

#############################################################################
# Step 3: Build the student's libraries
#############################################################################
echo ""
echo -e "${YELLOW}[Step 3/4]${NC} Building student's libraries..."

cd "${BUILD_DIR}"

echo "Running cmake..."
if ! cmake "${PROJECT_ROOT}" > cmake_output.log 2>&1; then
    echo -e "${RED}✗ ERROR: cmake configuration failed${NC}"
    echo "See ${BUILD_DIR}/cmake_output.log for details"
    cat cmake_output.log
    exit 1
fi
echo -e "${GREEN}✓${NC} CMake configuration successful"

echo "Running make..."
if ! make > make_output.log 2>&1; then
    echo -e "${RED}✗ ERROR: make build failed${NC}"
    echo "See ${BUILD_DIR}/make_output.log for details"
    cat make_output.log
    exit 1
fi
echo -e "${GREEN}✓${NC} Build successful"

#############################################################################
# Step 4: Build and run the grader
#############################################################################
echo ""
echo -e "${YELLOW}[Step 4/4]${NC} Building grader..."

# Check if grader source exists
if [ ! -f "${SCRIPT_DIR}/grader.cu" ]; then
    echo -e "${RED}✗ ERROR: grader.cu not found in autograder directory${NC}"
    exit 1
fi

cd "${BUILD_DIR}"

# Compile grader with the student's libraries
echo "Compiling grader..."
if ! nvcc -o grader "${SCRIPT_DIR}/grader.cu" \
    -I"${PROJECT_ROOT}/include" \
    -L"${BUILD_DIR}/src" \
    -lcudaLib -lcpuLib \
    -lcurand \
    -Xcompiler -Wall \
    > grader_compile.log 2>&1; then
    echo -e "${RED}✗ ERROR: grader compilation failed${NC}"
    echo "See ${BUILD_DIR}/grader_compile.log for details"
    cat grader_compile.log
    exit 1
fi
echo -e "${GREEN}✓${NC} Grader compiled successfully"

# Verify grader executable exists
if [ ! -f "${GRADER_EXECUTABLE}" ]; then
    echo -e "${RED}✗ ERROR: grader executable not found after compilation${NC}"
    exit 1
fi

echo ""
echo "Running grader..."
echo ""

# Run the grader and capture exit code
cd "${BUILD_DIR}"
set +e  # Don't exit on error for this command
"${GRADER_EXECUTABLE}"
GRADER_EXIT_CODE=$?
set -e

echo ""
#############################################################################
# Final Result
#############################################################################
if [ $GRADER_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                     GRADING PASSED ✓                      ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
    exit 0
else
    echo -e "${RED}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║                     GRADING FAILED ✗                      ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}Please review the test failures above and fix your implementation.${NC}"
    exit 1
fi
