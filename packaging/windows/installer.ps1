param(
    [string]$EnvName = "dev-cpu",
    [string]$Variant = "CPU",
    [string]$Tag = "",
    [string]$OutputDir = "release-artifacts/nsis",
    [string]$NsiPath = "packaging/windows/installer.nsi",
    [string]$LicensePath = "packaging/windows/license.rtf"
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

function Get-ProductVersion {
    param([string]$TagName)

    if ($TagName -match '^v(\d+)\.(\d+)\.(\d+)$') {
        return "$($Matches[1]).$($Matches[2]).$($Matches[3]).0"
    }

    if ($TagName -match '^v(\d+)\.(\d+)\.(\d+)(alpha|beta|rc)(\d+)$') {
        $major = [int]$Matches[1]
        $minor = [int]$Matches[2]
        $patch = [int]$Matches[3]
        $channel = $Matches[4]
        $number = [int]$Matches[5]

        $offset = switch ($channel) {
            "alpha" { 0 }
            "beta" { 20 }
            "rc" { 40 }
            default { 0 }
        }

        return "$major.$minor.$patch.$($offset + $number)"
    }

    return "2.0.0.0"
}

$repoRoot = Resolve-Path "."
$envPrefix = Join-Path $repoRoot ".pixi\envs\$EnvName"
$exeDir = Join-Path $envPrefix "bin"
$runtimeBinDir = Join-Path $envPrefix "Library\bin"
$exePath = Join-Path $exeDir "SPONGE.exe"

if (-not (Test-Path $exePath)) {
    throw "SPONGE.exe not found at $exePath. Build the $EnvName environment first."
}

# Per-variant staging directory to allow parallel builds
$stageDir = Join-Path $OutputDir "stage-$($Variant.ToLower())"

# Prepare output and staging directories
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
if (Test-Path $stageDir) {
    Remove-Item -Recurse -Force $stageDir
}
New-Item -ItemType Directory -Force -Path $stageDir | Out-Null

# Stage SPONGE.exe
Copy-Item $exePath -Destination $stageDir

# Stage all DLLs (deduplicated)
$copiedDlls = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
foreach ($dllDir in @($exeDir, $runtimeBinDir)) {
    if (-not (Test-Path $dllDir)) {
        continue
    }

    Get-ChildItem -Path $dllDir -Filter "*.dll" -File | ForEach-Object {
        if ($copiedDlls.Add($_.Name)) {
            Copy-Item $_.FullName -Destination $stageDir
        }
    }
}

# Resolve paths
$tagLabel = if ($Tag) { $Tag } else { "dev" }
$productVersion = Get-ProductVersion $Tag
$variantUpper = $Variant.ToUpper()
$outputPath = Join-Path (Resolve-Path $OutputDir) "SPONGE-$variantUpper-$tagLabel-installer.exe"
$nsiFullPath = Join-Path $repoRoot $NsiPath
$licenseFullPath = Join-Path $repoRoot $LicensePath
$stageFullPath = Resolve-Path $stageDir

# Find makensis
$makensis = Get-Command "makensis" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source
if (-not $makensis) {
    $defaultPath = "${env:ProgramFiles(x86)}\NSIS\makensis.exe"
    if (Test-Path $defaultPath) {
        $makensis = $defaultPath
    } else {
        throw "NSIS not found. Install NSIS or ensure makensis is in PATH."
    }
}

# Ensure .nsi file has UTF-8 BOM (required by NSIS for Unicode)
$nsiContent = [System.IO.File]::ReadAllText($nsiFullPath, [System.Text.Encoding]::UTF8)
$utf8Bom = New-Object System.Text.UTF8Encoding $true
[System.IO.File]::WriteAllText($nsiFullPath, $nsiContent, $utf8Bom)

# Build installer
& $makensis `
    /DPRODUCT_VERSION="$productVersion" `
    /DVARIANT="$variantUpper" `
    /DSTAGE_DIR="$stageFullPath" `
    /DOUTPUT_PATH="$outputPath" `
    /DLICENSE_FILE="$licenseFullPath" `
    /V2 `
    $nsiFullPath

if ($LASTEXITCODE -ne 0) {
    throw "makensis failed with exit code $LASTEXITCODE."
}

Write-Host "Created installer: $outputPath"
