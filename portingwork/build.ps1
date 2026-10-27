# Build the kindi Vulkan engine on Windows for this laptop (Intel Meteor Lake iGPU).
# Toolchain found on this machine: Vulkan SDK 1.4.363.0 + MSVC BuildTools 2022 (14.44.35207)
# + Windows SDK 10.0.26100.0. nlohmann/json cloned to vendor\nlohmann-json.

$glsl = "C:\VulkanSDK\1.4.363.0\Bin\glslangValidator.exe"
if (-not (Test-Path $glsl)) { $glsl = "glslangValidator" }
Write-Host "compiling shaders with $glsl"
Get-ChildItem shaders\*.comp | ForEach-Object {
    $out = "shaders\$($_.BaseName).comp.spv"
    Remove-Item $out -ErrorAction SilentlyContinue   # never let a stale module survive a failed compile
    & $glsl -V --target-env vulkan1.1 -o $out $_.FullName 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $out)) { throw "shader failed: $($_.Name)" }
}
# Parameterized variants: rows-per-wave width, selected at runtime from the device's
# wave size (wave128 -> 1, wave64 -> 2, wave32 -> 4). MUST be compiled from the same
# source as the default build - stale variants silently keep old behaviour.
$variants = @{
    "g4lut2r"       = @(1, 2, 4)
    "g4lut3r"       = @(1, 2, 4)
    "g4lutMr"       = @(1, 2, 4)
    "g5ptqM2r"      = @(1, 2, 4)
    "g5ptq_byte3nr" = @(1, 2, 4)
    "g5ptq_byte3nr_silu" = @(1, 2, 4)
    "g5ptq_bytenr"  = @(1, 2, 4, 8)
    "g5ptq_pf2r"    = @(2)
}
foreach ($base in $variants.Keys) {
    foreach ($w in $variants[$base]) {
        $out = "shaders\$($base)$w.comp.spv"
        Remove-Item $out -ErrorAction SilentlyContinue
        & $glsl -V --target-env vulkan1.1 "-DRPW=$w" -o $out "shaders\$base.comp" 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path $out)) { throw "variant shader failed: $base RPW=$w" }
    }
}
Write-Host ("shaders: " + (Get-ChildItem shaders\*.comp.spv).Count + " SPIR-V modules")

$vs     = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207"
$winsdk = "C:\Program Files (x86)\Windows Kits\10"
$wver   = "10.0.26100.0"
$vksdk  = "C:\VulkanSDK\1.4.363.0"

$env:INCLUDE = "$vs\include;$winsdk\Include\$wver\ucrt;$winsdk\Include\$wver\shared;$winsdk\Include\$wver\um;$vksdk\Include"
$env:LIB     = "$vs\lib\x64;$winsdk\Lib\$wver\ucrt\x64;$winsdk\Lib\$wver\um\x64;$vksdk\Lib"
$env:PATH    = "$vs\bin\Hostx64\x64;$env:PATH"

New-Item -ItemType Directory -Force -Path bin | Out-Null
foreach ($src in @("kindi_vk_decode", "kindi_cpu", "kindi_vk_engine", "kindi_vk_full", "kindi_vk", "kindi_vk_llama")) {
    Write-Host "building $src"
    cl /O2 /std:c++17 /EHsc /nologo /I $vksdk\Include /I vendor\nlohmann-json\include "$src.cpp" `
       /link /LIBPATH:"$vksdk\Lib" vulkan-1.lib ws2_32.lib /OUT:"bin\$src.exe"
    if ($LASTEXITCODE -ne 0) { throw "build failed: $src" }
}
Write-Host "done - binaries in bin\"
