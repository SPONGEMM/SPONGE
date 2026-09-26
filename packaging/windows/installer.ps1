param(
    [string]$EnvName = "dev-cpu",
    [string]$Variant = "CPU",
    [string]$Tag = "",
    [string]$OutputDir = "release-artifacts/nsis",
    [string]$NsiPath = "packaging/windows/installer.nsi",
    [string]$LicensePath = "LICENSE"
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

function Get-ProductVersion {
    param([string]$TagName)

    if (-not $TagName) {
        return "0.0.0.0"
    }

    if ($TagName -cnotmatch '\Av(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:-(alpha|beta|rc)\.(0|[1-9][0-9]*))?\z') {
        throw "Unsupported release tag: $TagName"
    }

    $parts = foreach ($index in 1..3) {
        $value = 0
        if (-not [int]::TryParse($Matches[$index], [ref]$value) -or $value -gt 65535) {
            throw "Release version components must be in 0..65535: $TagName"
        }
        $value
    }

    # Disjoint prerelease ranges preserve alpha < beta < rc < stable.
    # The revision is a 16-bit value; reserve its maximum for stable releases.
    $revision = 65535
    if ($Matches[4]) {
        $channel = $Matches[4]
        $number = 0
        if (-not [int]::TryParse($Matches[5], [ref]$number) -or $number -gt 19999) {
            throw "Prerelease number must be in 0..19999: $TagName"
        }

        $offset = switch ($channel) {
            "alpha" { 0 }
            "beta" { 20000 }
            "rc" { 40000 }
        }
        $revision = $offset + $number
    }

    return "$($parts -join '.').$revision"
}

# Validate before staging files or creating output directories.
$productVersion = Get-ProductVersion $Tag
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
$displayVersion = if ($Tag) { $Tag.Substring(1) } else { "dev" }
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
    /DDISPLAY_VERSION="$displayVersion" `
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
