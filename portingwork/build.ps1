# Build the kindi Vulkan engine on Windows (RX 580 / GCN, no containers).

# ---- shaders ----
$glsl = "C:\VulkanSDK\1.4.363.0\Bin\glslangValidator.exe"
if (-not (Test-Path $glsl)) { $glsl = "glslangValidator" }
Write-Host "compiling shaders with $glsl"
Get-ChildItem shaders\*.comp | ForEach-Object {
    $out = "shaders\$($_.BaseName).comp.spv"
    & $glsl -V --target-env vulkan1.1 -o $out $_.FullName 2>&1 | Out-Null
    if (-not (Test-Path $out)) { throw "shader failed: $($_.Name)" }
}
Write-Host ("shaders: " + (Get-ChildItem shaders\*.comp.spv).Count + " SPIR-V modules")

# ---- engine ----
# Adjust these to your Vulkan SDK / MSVC / SDK versions.
$vksdk  = "C:\VulkanSDK\1.4.363.0"
$vs     = "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Tools\MSVC\14.44.35207"
$winsdk = "C:\Program Files (x86)\Windows Kits\10"
$wver   = "10.0.22621.0"

if (-not (Test-Path $vksdk))  { throw "set `$vksdk to your Vulkan SDK" }
if (-not (Test-Path $vs))     { throw "set `$vs to your MSVC toolset" }

$env:INCLUDE = "$vs\include;$winsdk\Include\$wver\ucrt;$winsdk\Include\$wver\shared;$winsdk\Include\$wver\um"
$env:LIB     = "$vs\lib\x64;$winsdk\Lib\$wver\ucrt\x64;$winsdk\Lib\$wver\um\x64"
$env:PATH    = "$vs\bin\Hostx64\x64;$env:PATH"

# nlohmann/json.hpp is needed by the tokenizer headers. Point -I at any copy, e.g.
#   git clone https://github.com/nlohmann/json vendor/nlohmann-json
$json = "vendor\nlohmann-json\include"
if (-not (Test-Path "$json\nlohmann\json.hpp")) {
    Write-Warning "no nlohmann/json.hpp at $json - set `$json"
}

New-Item -ItemType Directory -Force -Path bin | Out-Null
foreach ($src in @("kindi_vk_decode", "kindi_cpu", "kindi_vk_llama", "kindi_vk_engine", "kindi_vk_full", "kindi_vk")) {
    Write-Host "building $src"
    cl /O2 /std:c++17 /EHsc /nologo /I "$vksdk\Include" /I $json "$src.cpp" `
       /link /LIBPATH:"$vksdk\Lib" vulkan-1.lib /OUT:"bin\$src.exe"
    if ($LASTEXITCODE -ne 0) { throw "build failed: $src" }
}
Write-Host "done - binaries in bin\"
