param(
    [string]$EnvName = "dev-cpu",
    [string]$Tag = "",
    [string]$OutputDir = "release-artifacts/msi",
    [string]$StageDir = "release-artifacts/msi/stage",
    [string]$TemplatePath = "packaging/windows/cpu.wxs"
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

function New-WixGuid {
    return ([guid]::NewGuid().ToString().ToUpper())
}

$repoRoot = Resolve-Path "."
$envPrefix = Join-Path $repoRoot ".pixi\envs\$EnvName"
$exeDir = Join-Path $envPrefix "bin"
$runtimeBinDir = Join-Path $envPrefix "Library\bin"
$exePath = Join-Path $exeDir "SPONGE.exe"

if (-not (Test-Path $exePath)) {
    throw "SPONGE.exe not found at $exePath. Build the Windows CPU environment first."
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
if (Test-Path $StageDir) {
    Remove-Item -Recurse -Force $StageDir
}
New-Item -ItemType Directory -Force -Path $StageDir | Out-Null

Copy-Item $exePath -Destination $StageDir

$copiedDlls = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
foreach ($dllDir in @($exeDir, $runtimeBinDir)) {
    if (-not (Test-Path $dllDir)) {
        continue
    }

    Get-ChildItem -Path $dllDir -Filter "*.dll" -File | ForEach-Object {
        if ($copiedDlls.Add($_.Name)) {
            Copy-Item $_.FullName -Destination $StageDir
        }
    }
}

$template = Get-Content $TemplatePath -Raw
$componentLines = New-Object System.Collections.Generic.List[string]
$refLines = New-Object System.Collections.Generic.List[string]

Get-ChildItem -Path $StageDir -File | Sort-Object Name | ForEach-Object {
    $componentId = "cmp_" + [IO.Path]::GetFileNameWithoutExtension($_.Name).Replace("-", "_").Replace(".", "_")
    $fileId = "fil_" + [IO.Path]::GetFileNameWithoutExtension($_.Name).Replace("-", "_").Replace(".", "_")
    $guid = New-WixGuid
    $name = $_.Name
    $source = $_.FullName

    $componentLines.Add("      <Component Id=""$componentId"" Guid=""$guid"" Win64=""yes"">")
    $componentLines.Add("        <File Id=""$fileId"" Source=""$source"" KeyPath=""yes"" />")
    $componentLines.Add("      </Component>")
    $refLines.Add("      <ComponentRef Id=""$componentId"" />")
}

$rendered = $template.Replace("{{FILE_COMPONENTS}}", ($componentLines -join "`r`n"))
$rendered = $rendered.Replace("{{COMPONENT_REFS}}", ($refLines -join "`r`n"))

$tagLabel = if ($Tag) { $Tag } else { "dev" }
$productVersion = Get-ProductVersion $Tag
$rendered = $rendered.Replace("{{PRODUCT_VERSION}}", $productVersion)
$wxsPath = Join-Path $OutputDir "cpu.generated.wxs"
$wixObjPath = Join-Path $OutputDir "cpu.wixobj"
$msiPath = Join-Path $OutputDir "SPONGE-CPU-$tagLabel.msi"

Set-Content -Path $wxsPath -Value $rendered -Encoding UTF8

$wixRoot = $env:WIX
if (-not $wixRoot) {
    $defaultWixRoot = "${env:ProgramFiles(x86)}\WiX Toolset v3.11\"
    if (Test-Path $defaultWixRoot) {
        $wixRoot = $defaultWixRoot
    }
}

$wixBin = Join-Path $wixRoot "bin"
if (-not (Test-Path $wixBin)) {
    throw "WiX Toolset not found. Expected WIX environment variable or default WiX v3 install path."
}

& (Join-Path $wixBin "candle.exe") `
    -out $wixObjPath `
    $wxsPath

& (Join-Path $wixBin "light.exe") `
    -ext WixUIExtension `
    -out $msiPath `
    $wixObjPath

Write-Host "Created MSI: $msiPath"
