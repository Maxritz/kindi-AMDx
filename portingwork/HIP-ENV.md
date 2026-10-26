# HIP environment on this machine

ROCm 7.16 at `C:\Rocm10`, target `gfx1031` (RX 6700 XT, Navi 22, RDNA2).

Dot-source `hip-env.ps1` before building or running anything:

```powershell
. .\hip-env.ps1
```

## The three things that cost time

### 1. hipcc cannot find the device library

```
clang: error: cannot find ROCm device library; provide its path via '--rocm-path'
        or '--rocm-device-lib-path', or pass '-nogpulib'
```

`--rocm-path=C:\Rocm10` does **not** fix it. The bitcode is at
`C:\Rocm10\lib\llvm\amdgcn\bitcode`, which is not where clang looks by default on this
install, so the path has to be given explicitly:

```powershell
hipcc -O3 --offload-arch=gfx1031 `
      --rocm-device-lib-path=C:\Rocm10\lib\llvm\amdgcn\bitcode `
      kernel.cpp -o kernel.exe
```

`hip-env.ps1` exports it as `$env:HIP_DEVICE_LIB`.

### 2. "no ROCm-capable device is detected" from your own exe

The symptom, with the card perfectly healthy:

```
hipGetDeviceCount -> no ROCm-capable device is detected (count=0)
```

while `C:\Rocm10\bin\hipInfo.exe` reports the RX 6700 XT with full detail, from the same
shell, with the same PATH.

The cause: `amdhip64_7.dll` depends on `rocm_kpack.dll` and `amd_comgr.dll`, and Windows
resolves those from the **exe's own directory** before PATH. When the exe lives elsewhere and
the ROCm bin dir is only on PATH, the dependency chain does not resolve and the runtime
silently reports no device rather than a load error.

Three ways to fix it, all verified:

| fix | how |
|---|---|
| copy the runtime next to the exe | `amdhip64_7.dll`, `rocm_kpack.dll`, `amd_comgr.dll` |
| launch from the ROCm bin dir | `Copy-Item app.exe C:\Rocm10\bin\; cd C:\Rocm10\bin; .\app.exe` |
| set `ROCM_PATH` **and** copy | still needs the copy; `ROCM_PATH` alone is not enough |

Copying is what a shipped bundle has to do anyway, so that is the recommended route.

Note `amd_comgr.dll` is 117 MB, so a bundle carries about 133 MB of runtime before any
weights.

### 3. rocblas: sgemv is unimplemented, sgemm works, and it needs Tensile's path

Three separate problems, all measured:

**It looks for `./rocblas/library/` next to the exe.** Without `ROCBLAS_TENSILE_LIBPATH` it
fails with:

```
rocBLAS error: Cannot read <exedir>/rocblas/library/TensileLibrary.dat: No such file or
directory for GPU arch : gfx1031
```

Fix: `ROCBLAS_TENSILE_LIBPATH=C:\Rocm10\bin\rocblas\library` (159 MB, 994 files, so set
the variable rather than copying it).

**`rocblas_sgemv` is not implemented for gfx1031.** It returns status 6 and throws
`hipErrorInvalidKernelFile`, even with the path set. `rocblas_sgemm` works fine. A GEMV is a
GEMM with `n=1`, so use that:

```cpp
rocblas_sgemm(h, rocblas_operation_transpose, rocblas_operation_none,
              R, 1, C, &alpha, A, C, x, C, &beta, y, R);
```

with `A` stored row-major `R x C` (which is `C x R` column-major with `lda=C`, hence the
transpose).

**The first call costs about 390 ms.** That is Tensile loading its kernel library, not the
GEMM. It is one-time per process; every later call is fast. Worth knowing before reading a
trace that shows a single enormous sgemm.

### 4. `.kpack`

`rocblas.dll` reads its precompiled kernels from `$env:ROCM_PATH\.kpack`. This install has
`blas_lib_gfx1031.kpack` there, so `ROCM_PATH=C:\Rocm10` must be set even when the DLLs are
copied locally. (The HIP backend does not call rocblas - see HIP-PORT.md - but this matters
for anything that does.)

## Verified working

```
device      : AMD Radeon RX 6700 XT
gcnArchName : gfx1031
CU          : 20 WGP (40 CU)
clock       : 2.42 GHz
warpSize    : 32
VRAM        : 11.98 GB, 192-bit, 96 MB Infinity Cache
LDS         : 64 KB/workgroup, 128 KB/WGP, 64 banks
saxpy       : y[0] = 5.0  OK
```

## wave64 is not an option

`-mwavefrontsize64` crashes the linker on this ROCm build:

```
Stack dump:
0.  Program arguments: ld.lld ... -plugin-opt=mcpu=gfx1031 ...
2.  Running pass 'Branch relaxation pass' on function '@...'
```

That is an LLVM bug, not a misconfiguration. It does not matter, because per the RDNA2 ISA
section 2.1 wave64 is issued as two wave32 passes anyway, so there is nothing to gain. Use
wave32.

## Instruction probe results

Compiled in isolation, disassembled with `llvm-objdump`:

| construct | result |
|---|---|
| `__builtin_amdgcn_sdot4` -> `v_dot4c_i32_i8` | **works** |
| `__builtin_amdgcn_sdot8` -> `v_dot8_i32_i4` | **works** |
| `__builtin_amdgcn_fdot2` | compiles; needs a `_Float16` vector argument, not `float` |
| `__builtin_amdgcn_wmma_*` | **not on RDNA2** (RDNA3) |

## Host/device pitfalls hit while building the probes

- `warpSize` is a `__device__` builtin. Reading it from host code fails to compile
  ("reference to `__device__` function"). Capture it from `hipGetDeviceProperties`.
- A weight int32 holds **four bytes = 16 trits**, but `dp4a` only multiplies four int8 lanes.
  One `dp4a` covers four *columns*, so the loop must stride **bytes**, not words. Getting this
  wrong produces plausible-looking numbers that fail the reference.
