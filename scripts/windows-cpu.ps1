param(
    [ValidateSet('configure', 'compile', 'test')]
    [string]$Action = 'configure',
    [string]$Parallel = 'none',
    [int]$Jobs = 4,
    [string]$DependencyPrefix = $env:CONDA_PREFIX,
    [string]$EnvironmentName = $env:PIXI_ENVIRONMENT_NAME,
    [switch]$Activated,
    [switch]$BuildTests
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
if (-not $DependencyPrefix -or -not $EnvironmentName) {
    throw 'Run this script through pixi run -e <cpu environment>.'
}

if (-not $Activated) {
    # Keep Intel's compiler/runtime packages separate from the LLVM JIT packages.
    $manifest = Join-Path $projectRoot 'tools/oneapi/pixi.toml'
    $arguments = @('run', '--locked', '--manifest-path', $manifest,
        'powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $PSCommandPath,
        '-Activated', '-Action', $Action, '-Parallel', $Parallel, '-Jobs', $Jobs,
        '-DependencyPrefix', $DependencyPrefix, '-EnvironmentName', $EnvironmentName)
    if ($BuildTests) { $arguments += '-BuildTests' }
    $pixi = $env:PIXI_EXE
    if (-not $pixi) { $pixi = (Get-Command pixi -ErrorAction Stop).Source }
    & $pixi @arguments
    exit $LASTEXITCODE
}

$toolset = $env:VCToolsVersion
if (-not $toolset -or [version]$toolset.TrimEnd('\') -lt [version]'14.44') {
    throw "Windows CPU requires VS 2022 C++ Build Tools 14.44 or newer; selected: '$toolset'. Update the C++ tools in Visual Studio Installer (updating the VC runtime alone is insufficient)."
}
$compiler = (Get-Command icx-cl.exe -ErrorAction Stop).Source
$dependencyLibrary = Join-Path $DependencyPrefix 'Library'
$env:INCLUDE = "$dependencyLibrary\include;$env:INCLUDE"
$env:LIB = "$dependencyLibrary\lib;$env:LIB"
$env:PATH = "$DependencyPrefix;$dependencyLibrary\bin;$env:PATH"
$env:PYTHONUTF8 = '1'
$cmake = Join-Path $dependencyLibrary 'bin/cmake.exe'
$buildDir = Join-Path $projectRoot "build-$EnvironmentName-oneapi"
Write-Host "Intel compiler: $compiler"
Write-Host "VS toolset: $toolset; dependency prefix: $DependencyPrefix"

switch ($Action) {
    'configure' {
        $options = @('-S', $projectRoot, '-B', $buildDir, '-G', 'Ninja',
            "-DPARALLEL=$Parallel", "-DCMAKE_INSTALL_PREFIX=$DependencyPrefix",
            "-DCMAKE_PREFIX_PATH=$dependencyLibrary", "-DCMAKE_C_COMPILER=$compiler",
            "-DCMAKE_CXX_COMPILER=$compiler",
            "-DCMAKE_MAKE_PROGRAM=$dependencyLibrary/bin/ninja.exe")
        if (Test-Path "$DependencyPrefix/python.exe") {
            $options += "-DPython3_EXECUTABLE=$DependencyPrefix/python.exe"
        }
        $options += "-DSPONGE_BUILD_TESTS=$($BuildTests.IsPresent.ToString().ToUpperInvariant())"
        & $cmake @options
    }
    'compile' { & $cmake --build $buildDir --target install --parallel $Jobs }
    'test' {
        & "$dependencyLibrary/bin/ctest.exe" --test-dir $buildDir --output-on-failure
    }
}
exit $LASTEXITCODE
