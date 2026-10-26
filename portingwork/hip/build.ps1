# Build the HIP engine for the RX 6700 XT (gfx1031).
#
# Requires ROCm 7.16 at C:\Rocm10. Three things that are easy to get wrong, all documented
# in HIP-ENV.md:
#
#   1. hipcc needs --rocm-device-lib-path explicitly; --rocm-path does not work.
#   2. The exe must find amdhip64_7.dll's siblings from its OWN directory, so the runtime
#      is copied next to it rather than relying on PATH.
#   3. rocblas reads its Tensile kernels from ROCBLAS_TENSILE_LIBPATH (or ./rocblas/library
#      next to the exe). It is set here rather than copied, because the directory is 159 MB.

$ErrorActionPreference = 'Stop'
$rocm = 'C:\Rocm10'
$arch = 'gfx1031'
$devlib = "$rocm\lib\llvm\amdgcn\bitcode"

if (-not (Test-Path "$rocm\bin\hipcc.exe")) { throw "hipcc not found at $rocm\bin" }
if (-not (Test-Path $devlib)) { throw "device library not found at $devlib" }

$env:ROCM_PATH = $rocm
$env:HIP_PATH = $rocm
$env:HIP_PLATFORM = 'amd'
$env:ROCBLAS_TENSILE_LIBPATH = "$rocm\bin\rocblas\library"
$env:PATH = "$rocm\bin;$rocm\lib\llvm\bin;$env:PATH"

Write-Host "building kindi_hip for $arch"
# hipcc writes an unrelated linker warning to stderr, and with ErrorActionPreference=Stop
# PowerShell turns any native stderr into a terminating error. Relax it for the call and
# judge success by the exit code, which is the thing that actually matters.
$ErrorActionPreference = 'Continue'
& "$rocm\bin\hipcc.exe" -O3 -std=c++17 --offload-arch=$arch `
    "--rocm-device-lib-path=$devlib" `
    "-I$rocm\include" `
    kindi_hip.cpp -o kindi_hip.exe `
    "-L$rocm\lib" -lrocblas 2>&1 | Where-Object { $_ -notmatch 'ld-path|unused during compilation' }
$code = $LASTEXITCODE
$ErrorActionPreference = 'Stop'
if ($code -ne 0) { throw "hipcc failed with $code" }

# The runtime DLLs have to sit next to the exe. amdhip64_7.dll resolves rocm_kpack.dll and
# amd_comgr.dll from the exe directory before PATH, and without them hipGetDeviceCount
# silently reports "no ROCm-capable device is detected" rather than a load error.
foreach ($d in 'amdhip64_7.dll', 'rocm_kpack.dll', 'amd_comgr.dll', 'rocblas.dll') {
    Copy-Item "$rocm\bin\$d" . -Force
}

Write-Host "built kindi_hip.exe"
Write-Host "run:  .\kindi_hip.exe --model <file.gguf> --selftest"
