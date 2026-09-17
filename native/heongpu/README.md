# HEonGPU BFV backend

This backend accelerates the one-time encrypted Gram-matrix computation used to prepare Proposed's virtual groups. It leaves the training interface and group-cache format unchanged.

## Requirements

Install these dependencies separately:

- HEonGPU 1.1.3, including its exported CMake package.
- NTL headers and libraries.
- A compatible NVIDIA GPU, driver and CUDA toolkit.
- CMake 3.26.4 or newer and a C++17 compiler with OpenMP support.

The repository contains the adapter source, not the dependency libraries or a compiled executable.

## Build

Follow the [complete dependency installation and build steps](../../README.md#build-heongpu)
in the main README. Run them on the GPU server, using the same `hflm3`
environment for compilation, Python training and tests.

The adapter script accepts these overrides:

| Variable | Default or purpose |
| --- | --- |
| `CMAKE_BIN` | `cmake` found on PATH |
| `CUDA_ROOT` | Prefix inferred from `nvcc` on PATH |
| `HEONGPU_ROOT` | `~/.local/heongpu-v1.1.3` |
| `NTL_ROOT` | Active Conda environment prefix |
| `CUDA_ARCH` | First GPU's compute capability from `nvidia-smi`, e.g. `89` for RTX 40 series |
| `BUILD_JOBS` | Parallel compilation jobs; default `4` |
| `BUILD_DIR` | Adapter output directory; default `build/heongpu/` |

Missing tools, nonexistent prefixes and `/path/to/...` placeholders are rejected
with an explanatory error. Activate `hflm3` before running the script.
The main README explicitly sets `HEONGPU_ROOT` to
`~/.local/heongpu-v1.1.3-hflm3`, overriding the script default. It builds in
`build/heongpu-hflm3/` and copies the executable to the default
`build/heongpu/private_gram_gpu` location to avoid old CMake cache entries.

## Run

Follow the [main README](../../README.md#quick-start-fashion-mnist) to prepare client data and invoke `scripts/prepare_private_virtual_groups.py --privacy_backend heongpu_gpu`.

The default executable is `build/heongpu/private_gram_gpu`. For a custom build location, pass `--gpu_executable /path/to/private_gram_gpu` or set `HFLM_HEONGPU_GRAM_BIN`.

Set the runtime library search path when necessary:

```bash
export LD_LIBRARY_PATH="$NTL_ROOT/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
```

The preparation CLI selects polynomial degree `8192` and plaintext modulus `33832961` by default for HEonGPU. It validates the recovered Gram matrix against the plaintext reference before saving groups. Existing matching caches are reused unless `--force` is supplied; preserve any historical cache before replacing it.

## Execution scope

The native executable emulates client encryption, encrypted evaluation and selector decryption in one process. This is an experiment harness, not a distributed service deployment. It reports preprocessing stage timings; these should be distinguished from model-training time and simulated communication latency.

Python component tests do not compile this backend. After building it, run `python -m pytest tests/test_private_gram_gpu.py -q` for a small native Gram exactness check, then run group preparation with the intended data. The native test is skipped when the executable is absent.
