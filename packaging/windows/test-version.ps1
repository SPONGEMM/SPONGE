param([switch]$BuildInstaller)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$installerPath = Join-Path $PSScriptRoot "installer.ps1"

# Load only the version function, without running the packaging entry point.
$parseErrors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
    $installerPath, [ref]$null, [ref]$parseErrors
)
if ($parseErrors.Count) { throw "Installer parse errors: $parseErrors" }
$function = $ast.Find({
    param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
        $node.Name -eq "Get-ProductVersion"
}, $false)
if (-not $function) { throw "Get-ProductVersion not found" }
. ([scriptblock]::Create($function.Extent.Text))

$cases = [ordered]@{
    "" = "0.0.0.0"
    "v0.0.0" = "0.0.0.65535"
    "v2.0.0-alpha.0" = "2.0.0.0"
    "v2.0.0-alpha.21" = "2.0.0.21"
    "v2.0.0-alpha.19999" = "2.0.0.19999"
    "v2.0.0-beta.0" = "2.0.0.20000"
    "v2.0.0-beta.1" = "2.0.0.20001"
    "v2.0.0-beta.2" = "2.0.0.20002"
    "v2.0.0-beta.19999" = "2.0.0.39999"
    "v2.0.0-rc.0" = "2.0.0.40000"
    "v2.0.0-rc.19999" = "2.0.0.59999"
    "v2.0.0" = "2.0.0.65535"
    "v2.0.1-alpha.0" = "2.0.1.0"
    "v2.1.0-alpha.0" = "2.1.0.0"
    "v3.0.0-alpha.0" = "3.0.0.0"
    "v65535.65535.65535" = "65535.65535.65535.65535"
}
$previous = $null
foreach ($case in $cases.GetEnumerator()) {
    $actual = Get-ProductVersion $case.Key
    if ($actual -cne $case.Value) {
        throw "$($case.Key): expected $($case.Value), got $actual"
    }
    $version = [version]$actual
    if ($null -ne $previous -and $version -le $previous) {
        throw "Version ordering failed: $previous >= $version"
    }
    $previous = $version
}

$invalidTags = @(
    "v65536.0.0", "v0.65536.0", "v0.0.65536",
    "v65536.0.0-beta.1", "v0.65536.0-rc.1", "v0.0.65536-alpha.1",
    "v999999999999999999999.0.0", "v2.0.0-beta.999999999999999999999",
    "v2.0.0-alpha.20000", "v2.0.0-beta.20000", "v2.0.0-rc.20000",
    "v2.0.0-beta.-1", "v2.0.0-beta.01", "v02.0.0", "v2.00.0", "v2.0.00",
    "2.0.0", "v2.0", "v2.0.0beta2", "v2.0.0-preview.1", "v2.0.0+build.1",
    "V2.0.0", "v2.0.0-BETA.1", " v2.0.0", "v2.0.0`n"
)
foreach ($tag in $invalidTags) {
    $rejected = $false
    try { $null = Get-ProductVersion $tag } catch { $rejected = $true }
    if (-not $rejected) { throw "Invalid tag was accepted: $tag" }
}
Write-Host "Passed $($cases.Count) version cases, $($cases.Count - 1) ordering checks, and $($invalidTags.Count) invalid tags."

if ($BuildInstaller) {
    # Exercise the real packaging script using an inert payload; no SPONGE build needed.
    $tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ([guid]::NewGuid().ToString())
    $payloadDir = Join-Path $tempRoot ".pixi/envs/version-test/bin"
    New-Item -ItemType Directory -Path $payloadDir -Force | Out-Null
    Set-Content -Path (Join-Path $payloadDir "SPONGE.exe") -Value "version test payload"
    Copy-Item (Join-Path $PSScriptRoot "installer.nsi") $tempRoot
    $licensePath = (Resolve-Path (Join-Path $PSScriptRoot "../../LICENSE")).Path
    Push-Location $tempRoot
    try {
        foreach ($tag in @("v2.0.0-alpha.19999", "v2.0.0-beta.2", "v2.0.0-rc.19999", "v2.0.0", "v2.0.1-alpha.0")) {
            & $installerPath -EnvName "version-test" -Variant CPU -Tag $tag `
                -NsiPath "installer.nsi" -LicensePath $licensePath
            $exe = Join-Path $tempRoot "release-artifacts/nsis/SPONGE-CPU-$tag-installer.exe"
            $info = [System.Diagnostics.FileVersionInfo]::GetVersionInfo($exe)
            $fixedVersion = "$($info.FileMajorPart).$($info.FileMinorPart).$($info.FileBuildPart).$($info.FilePrivatePart)"
            $fixedProductVersion = "$($info.ProductMajorPart).$($info.ProductMinorPart).$($info.ProductBuildPart).$($info.ProductPrivatePart)"
            if ($fixedVersion -ne $cases[$tag] -or $fixedProductVersion -ne $cases[$tag] -or
                $info.FileVersion -ne $cases[$tag] -or $info.ProductVersion -ne $tag.Substring(1)) {
                throw "Installer version metadata mismatch for $tag"
            }
        }
        Write-Host "Passed 5 NSIS installer metadata checks."
    } finally {
        Pop-Location
        Remove-Item -Recurse -Force $tempRoot
    }
}
