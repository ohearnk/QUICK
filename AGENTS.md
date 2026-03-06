# AGENTS.md — QUICK Quantum Chemistry Package

## Overview

QUICK is a GPU-accelerated quantum chemistry package written primarily in Fortran 90/95,
with C/C++ and CUDA/HIP code for GPU support. The codebase uses two parallel build systems:
a legacy `configure`+`make` system and a modern CMake system.

---

## Build Commands

### CMake Build (recommended)

```bash
mkdir build && cd build
cmake .. -DCOMPILER=GNU -DENABLEF=FALSE -DCMAKE_INSTALL_PREFIX=$PWD/../install
cmake --build . --parallel $(nproc)   # Linux: nproc; macOS: sysctl -n hw.logicalcpu
cmake --install .
source ../install/quick.rc   # sets QUICK_BASIS, PATH, LD_LIBRARY_PATH, LIBRARY_PATH
```

> **macOS + Homebrew GCC note:** On macOS, `/usr/bin/gcc` is an Apple Clang shim, so
> `-DCOMPILER=GNU` will be rejected by the build system's compiler identity check.
> Use `-DCOMPILER=AUTO` and pass the real compiler paths explicitly:
> ```bash
> cmake .. -DCOMPILER=AUTO -DENABLEF=FALSE \
>   -DCMAKE_INSTALL_PREFIX=$PWD/../install \
>   -DCMAKE_C_COMPILER=/opt/homebrew/bin/gcc-15 \
>   -DCMAKE_CXX_COMPILER=/opt/homebrew/bin/g++-15 \
>   -DCMAKE_Fortran_COMPILER=/opt/homebrew/bin/gfortran
> cmake --build . --parallel $(sysctl -n hw.logicalcpu)
> cmake --install .
> source ../install/quick.rc
> ```
> Adjust `gcc-15`/`g++-15` to match the version installed by Homebrew
> (`ls /opt/homebrew/bin/gcc-*`).

**Key CMake options:**

| Option | Values | Description |
|---|---|---|
| `-DCOMPILER` | `GNU`, `CLANG`, `INTELLLVM`, `ONEAPI`, `PGI`, `AUTO` | Compiler family |
| `-DENABLEF` | `TRUE`/`FALSE` (default: `FALSE`) | Enable f-function ERIs (expensive) |
| `-DCMAKE_BUILD_TYPE` | `Debug`, `Release` | Build type |
| `-DMPI` | `TRUE` | Enable MPI parallel build |
| `-DCUDA` | `TRUE` | Enable NVIDIA GPU build |
| `-DQUICK_USER_ARCH` | `volta`, `ampere`, etc. | CUDA GPU architecture |
| `-DWARNINGS` | `TRUE` | Enable compiler warnings |

See `CMake-Options.md` for the full list.

### Legacy Configure+Make Build

```bash
./configure --serial --enablef --shared --prefix $PWD/install gnu
make -j$(nproc) all install
source install/quick.rc   # sets QUICK_BASIS, PATH, LD_LIBRARY_PATH, LIBRARY_PATH
```

MPI variant: replace `--serial` with `--mpi`.
CUDA variant: replace `--serial` with `--cuda --arch volta`.

---

## Test Commands

Tests require QUICK to be **installed** first. The `runtest` script lives in the install
directory (or can be run from the repo root as `tools/runtest`). `QUICK_BASIS` must be set
(done automatically by `source install/quick.rc`).

### Run the full test suite

```bash
cd install
./runtest --serial --full        # 181 CPU tests
```

### Run the short test suite (CI default)

```bash
./runtest --serial               # short set (32 tests)
```

### Run a single test

QUICK has no built-in single-test flag. Run the executable directly:

```bash
export QUICK_BASIS=/path/to/install/basis
/path/to/install/bin/quick test/ene_H2O_rhf_sto3g.in
```

Compare output numerically with the saved reference:

```bash
awk -f test/ndiff.awk test/saved/ene_H2O_rhf_sto3g.out ene_H2O_rhf_sto3g.out
```

### Run tests by category

```bash
./runtest --serial --ene    # energy tests only
./runtest --serial --grad   # gradient tests only
./runtest --serial --opt    # geometry optimization tests only
./runtest --serial --api    # API tests only
./runtest --serial --esp    # ESP tests only
./runtest --serial --rw     # restart/checkpoint tests only
```

### MPI tests

```bash
DO_PARALLEL="mpirun -np 2" ./runtest --mpi --full
```

### Useful environment variables

```bash
QUICK_BASIS            # path to basis set directory (required)
DO_PARALLEL            # MPI launcher, e.g. "mpirun -np 2"
CUDA_VISIBLE_DEVICES   # GPU device IDs for CUDA tests
PARALLEL_TEST_COUNT    # number of tests to run in parallel (requires GNU parallel)
```

---

## Code Style Guidelines

### Languages

- **Fortran 90/95** — primary language (all SCF, DFT, gradients, MPI, API code)
- **C/C++** — utility subroutines, octree/grid packer, timing code
- **CUDA C/C++** — NVIDIA GPU kernels (`src/gpu/cuda/`)
- **HIP C/C++** — AMD GPU kernels (`src/gpu/hip/`)

### Fortran Naming Conventions

- **Modules:** `quick_<component>_module` — file named `quick_<component>_module.f90`
  - Examples: `quick_method_module`, `quick_basis_module`, `quick_exception_module`
- **Types:** `quick_<component>_type` — e.g., `quick_method_type`, `gpu_calculated_type`
- **Module instances (singletons):** lowercase — e.g., `quick_method`, `quick_molspec`
- **Subroutines/Functions:** `camelCase` or `PascalCase` for legacy code (`getEnergy`, `PrtAct`);
  newer code uses `snake_case` (`raise_exception`, `form_dft_grid`)
- **Variables:** lowercase, short abbreviations — e.g., `natom`, `nbasis`, `ierr`
- **Constants/Parameters:** `ALL_CAPS` — e.g., `PI`, `BOHR`, `OUTFILEHANDLE`
- **Logical flags in types:** `camelCase` — e.g., `quick_method%HF`, `quick_method%analGrad`
- **Preprocessor macros:** `ALL_CAPS` — e.g., `RECORD_TIME`, `MPIV`, `ENABLEF`, `DEBUG`

### C/C++/CUDA Naming Conventions

- **Structs/Types:** `snake_case` with `_type` suffix — e.g., `gpu_scratch`, `gpu_timer_type`
- **Functions:** `snake_case` or `camelCase` — e.g., `get2e`, `upload_sim_to_constant_oei`
- **Macros:** `ALL_CAPS` — e.g., `LOC2`, `LOC3`, `SQR`, `VDIM1`
- **Fortran-callable C functions:** use `extern "C"` with trailing underscore —
  e.g., `gpu_set_device_`, `gpu_upload_method_`

### File Naming

- Fortran modules: `quick_<component>_module.f90`
- Fortran subroutines (legacy): `PascalCase.f90` or `camelCase.f90`
- CUDA/C++ source: `snake_case.cu`, `snake_case.cpp`, `snake_case.h`
- GPU headers: `gpu_<purpose>.h` or `gpu_<component>_<type>.h`

### Indentation and Formatting

- **Fortran:** 3-space indentation (dominant convention); some legacy files use 2 or 4 spaces
- **C/CUDA:** 4-space indentation; opening `{` on same line as control structure
- **CMake:** 4-space (1-tab) indentation
- All Fortran code must use `implicit none` in every module/program/subroutine scope
- Long lines are allowed (compiler flag `-ffree-line-length-none` is set for GNU Fortran)
- The `tools/amindent` utility can be used to normalize Fortran indentation

### Include/Use Ordering

**Fortran `use` statements** — place at the top of the program unit, before `implicit none`:
1. Standard/external library modules (e.g., `use mpi`)
2. QUICK utility modules (e.g., `use allMod`, `use quick_constants_module`)
3. Functional QUICK modules (e.g., `use quick_method_module, only: quick_method`)

**Preprocessor include:** Every Fortran source file must include the project-wide utility
header near the very top (before the `module`/`subroutine` statement):
```fortran
#include "util.fh"
```
This header (`src/util/util.fh`) provides timing macros (`RECORD_TIME`, `START_TIME`,
`STOP_TIME`) and the `OUTFILEHANDLE` constant.

**C/CUDA headers:** Standard library headers first, then local project headers:
```c
#include <stdio.h>
#include <string>
#include "gpu.h"
#include "../gpu_common.h"
```

### Error Handling

- **Fortran:** Use an integer error flag `ierr` declared `intent(inout)` and passed through
  the call chain. Non-zero values indicate errors. Check and raise via:
  ```fortran
  call RaiseException(ierr)   ! from quick_exception_module
  ```
  This calls `quick_exit(OUTFILEHANDLE, 1)` on failure. Error codes are defined in
  `src/modules/quick_exception_module.f90` with `select case(ierr)` messages.
- **MPI code:** Wrap MPI-only logic in `#ifdef MPIV` / `#endif` preprocessor guards.
  Only the master process should do I/O:
  ```fortran
  if (master) write(OUTFILEHANDLE, ...) ...
  ```
- **CUDA:** Check CUDA API return codes explicitly within GPU utility functions.

### Comment Style

**Fortran:**
```fortran
! Single-line comment

!-----------------------------------------------------------------------!
! Section header banner (width ~72 chars)                               !
!_______________________________________________________________________!

subroutine foo(x, y)  ! inline comment after code
```

**C/CUDA:**
```c
// Single-line comment
/* Block comment for longer explanations */
```

Every source file should carry a copyright/license header (MPLv2):
```fortran
!
! (C) Copyright <year> QUICK contributors
! All rights reserved.
! ...
! This Source Code Form is subject to the terms of the Mozilla Public
! License, v. 2.0.
!
```

### MPI/GPU Portability

- CPU-only, MPI, CUDA, and HIP builds share the same source tree. Use preprocessor guards:
  - `#ifdef MPIV` for MPI-specific code
  - `#ifdef CUDA_MPIV` for CUDA+MPI code
  - `#ifdef OSHM_MPIV` for OpenSHMEM code
- GPU-callable device functions are annotated with `__device__` / `__global__` (CUDA) or
  the HIP equivalents.
- Fortran-to-GPU interoperability is done via `iso_c_binding` and `extern "C"` interfaces.

---

## Repository Layout (Key Paths)

```
src/modules/         Fortran modules (quick_*_module.f90) — all shared state/types
src/subs/            Fortran utility subroutines and C++ utility files
src/gpu/cuda/        NVIDIA CUDA kernels and GPU driver (gpu.cu)
src/gpu/hip/         AMD HIP kernels
src/octree/          C++ octree for DFT quadrature grids
src/libxc/           DFT XC functional library (libxc)
src/util/util.fh     Project-wide Fortran preprocessor header (MUST include)
test/                Regression test inputs (*.in) and references (saved/*.out)
test/testlist_full.txt     Full CPU test list (~181 tests)
test/testlist_short.txt    Short CPU test list
tools/runtest        Test runner script
basis/               Basis set data files
CMake-Options.md     Reference for all CMake build options
```

---

## CI

GitHub Actions workflows are in `.github/workflows/`:
- `build_test_serial.yml` — Serial CPU builds with GNU 10–14, Clang 17/18, IntelLLVM, NVHPC, macOS
- `build_test_mpi.yml` — MPI builds with OpenMPI/MPICH/Intel-MPI across the same compilers

Both run `./runtest --serial --full` (or `--mpi --full`) and upload test log artifacts.
