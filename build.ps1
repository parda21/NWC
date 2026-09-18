# Builds the CUDA library on Windows: build/nwc_ops.dll (used by the tests and scripts from the repository).
# Run from the project directory: .\build.ps1 [-Fatbin]   (-Fatbin: sm_80..sm_120 + PTX into nwc/lib for the wheel)
param([switch]$Fatbin)
$ErrorActionPreference = "Stop"
$R = $PSScriptRoot
$vc = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
$nvcc = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\nvcc.exe"
New-Item -ItemType Directory -Force "$R\build" | Out-Null
if ($Fatbin) {
    New-Item -ItemType Directory -Force "$R\nwc\lib" | Out-Null
    $gen = "-gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 " +
           "-gencode arch=compute_90,code=sm_90 -gencode arch=compute_120,code=sm_120 -gencode arch=compute_90,code=compute_90"
    $step = "`"$nvcc`" -O3 --shared -o nwc\lib\nwc_ops.dll csrc\nwc_ops.cu $gen"
} else {
    $step = "`"$nvcc`" -O3 -arch=sm_89 --shared -o build\nwc_ops.dll csrc\nwc_ops.cu"
}
cmd /c "`"$vc`" >nul 2>&1 && cd /d `"$R`" && $step"
if ($LASTEXITCODE -ne 0) { throw "build failed" }
Remove-Item "$R\build\*.obj", "$R\build\*.lib", "$R\build\*.exp", "$R\nwc\lib\*.lib", "$R\nwc\lib\*.exp" -ErrorAction SilentlyContinue
Write-Output "built:"; Get-ChildItem "$R\build\nwc_ops.dll", "$R\nwc\lib\nwc_ops.dll" -ErrorAction SilentlyContinue | Select-Object FullName, @{n='KB';e={[math]::Round($_.Length/1KB)}}
