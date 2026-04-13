[CmdletBinding()]
param(
    [string]$IntendedLabel = "unknown",

    [string]$BaseDir = "data/raw/live_buffer/openpose_session"
)

$ErrorActionPreference = "Stop"

function Get-RepoRoot {
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    return [System.IO.Path]::GetFullPath((Join-Path $scriptDir "..\.."))
}

function Sanitize-Label {
    param([string]$Label)

    if ([string]::IsNullOrWhiteSpace($Label)) {
        return "unknown"
    }

    $trimmed = $Label.Trim().ToLowerInvariant()
    $safe = [System.Text.RegularExpressions.Regex]::Replace($trimmed, "[^a-z0-9_\-]", "_")
    $safe = [System.Text.RegularExpressions.Regex]::Replace($safe, "_+", "_")
    $safe = $safe.Trim("_", "-")

    if ([string]::IsNullOrWhiteSpace($safe)) {
        return "unknown"
    }

    return $safe
}

$repoRoot = Get-RepoRoot
$resolvedBaseDir = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $BaseDir))
New-Item -ItemType Directory -Path $resolvedBaseDir -Force | Out-Null

$normalizedLabel = Sanitize-Label -Label $IntendedLabel
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$sessionName = "live_${normalizedLabel}_${timestamp}"
$sessionDir = Join-Path $resolvedBaseDir $sessionName

New-Item -ItemType Directory -Path $sessionDir -Force | Out-Null

$openposeCommand = @(
    'OpenPoseDemo.exe',
    '--write_json',
    '"' + $sessionDir + '"',
    '--display',
    '1',
    '--render_pose',
    '1'
) -join ' '

$replayCommand = @(
    'python -m src.inference.live_openpose_debug',
    '--json-dir',
    '"' + $sessionDir + '"',
    '--tracking-mode single_person',
    '--print-every-n 10',
    '--quiet-warmup',
    '--max-idle-polls 40',
    '--intended-label',
    $normalizedLabel
) -join ' '

$confidenceCommand = @(
    'python -m src.analysis.analyze_live_debug_confidence',
    '--csv',
    '"logs/inference/live_debug_YYYYMMDD_HHMMSS.csv"'
) -join ' '

Write-Host "Created session folder: $sessionDir" -ForegroundColor Green
Write-Host ""
Write-Host "OpenPose command to run next:" -ForegroundColor Cyan
Write-Host "  $openposeCommand"
Write-Host ""
Write-Host "Replay command to run after recording:" -ForegroundColor Cyan
Write-Host "  $replayCommand"
Write-Host ""
Write-Host "Confidence analysis command shape:" -ForegroundColor Cyan
Write-Host "  $confidenceCommand"
Write-Host ""
Write-Host "Tip: use the newest CSV from logs/inference as --csv input for confidence analysis." -ForegroundColor Yellow
