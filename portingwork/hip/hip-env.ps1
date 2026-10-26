# Set the HIP/ROCm environment for this machine (ROCm 7.16 at C:\Rocm10, gfx1031).
# Dot-source it:  . .\hip-env.ps1
#
# Why ROCM_PATH matters even though the exe links amdhip64_7.dll by name: the runtime
# loads its siblings from the exe directory, and rocblas reads its kernel packs from
# $env:ROCM_PATH\.kpack. Without it, hipGetDeviceCount returns
# "no ROCm-capable device is detected" and rocblas fails to find blas_lib_gfx1031.kpack.

$env:ROCM_PATH = 'C:\Rocm10'
$env:HIP_PATH  = 'C:\Rocm10'
$env:HIP_PLATFORM = 'amd'

# The device library is not where clang looks by default on this install, so hipcc needs
# it passed explicitly. Both are set here once instead of on every command line.
$env:HIP_DEVICE_LIB = 'C:\Rocm10\lib\llvm\amdgcn\bitcode'

$env:PATH = @(
    'C:\Rocm10\bin',
    'C:\Rocm10\lib\llvm\bin',
    $env:PATH
) -join ';'

# Target arch for this card.
$env:HIP_ARCH = 'gfx1031'

Write-Host "ROCM_PATH    $env:ROCM_PATH"
Write-Host "HIP_DEVICE_LIB $env:HIP_DEVICE_LIB"
Write-Host "HIP_ARCH     $env:HIP_ARCH"
Write-Host "hipcc        $((Get-Command hipcc -ErrorAction SilentlyContinue).Source)"
