<#
.SYNOPSIS
Beginner-friendly guided OpenPose helper for one full 8-gesture cycle.

.DESCRIPTION
This script records one complete gesture cycle for a single person/session.
It prompts once for person/session/video/duration, automatically chooses the
next available take label (take_###), then guides you through 8 gestures in a
fixed order. Before each gesture, it waits for ENTER and supports quitting.

For each gesture capture it uses the same timing behavior as
record_gesture_timed.ps1:
- countdown
- launch OpenPose
- wait until first JSON file appears
- then start timed recording
- stop OpenPose automatically after duration
#>

# ============================================================================
# EDITABLE DEFAULTS
# ----------------------------------------------------------------------------
# Update these paths for your machine. OpenPose must be installed outside this
# repository. This repo only stores the output data (JSON + optional video).
# ============================================================================
$OpenPoseRoot = "D:\Programs\OpenPose\openpose"
$ProjectRoot = "D:\Documentos\Python Projects\Avatar-Arcade-Game"

$DefaultPerson = "luis"
$DefaultSession = "s01"
$DefaultUseVideo = $true
$DefaultDurationSeconds = 3
$DefaultCountdownSeconds = 3

# Startup readiness settings:
# - Poll every 200 ms to detect first JSON frame quickly.
# - Give OpenPose up to 20 seconds to start producing output.
$CapturePollIntervalMs = 200
$CaptureStartupTimeoutSeconds = 20

# Fixed gesture order for one full cycle.
$GestureCycle = @(
    "attack_earth",
    "attack_fire",
    "attack_water",
    "attack_air",
    "defense_earth",
    "defense_fire",
    "defense_water",
    "defense_air"
)

# ============================================================================
# Helper function: read input with an optional default value.
# If user presses ENTER with no text, we use the default.
# ============================================================================
function Read-Value {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Prompt,

        [Parameter(Mandatory = $false)]
        [string]$Default = ""
    )

    if ([string]::IsNullOrWhiteSpace($Default)) {
        return Read-Host $Prompt
    }

    $value = Read-Host "$Prompt [$Default]"
    if ([string]::IsNullOrWhiteSpace($value)) {
        return $Default
    }

    return $value
}

# ============================================================================
# Helper function: determine the next take label across the full 8-gesture
# cycle for the selected person/session.
#
# Rules:
# - Look in data/raw/openpose_json/<gesture>/<person>/<session>/
# - Match folder names take_###
# - Use max number + 1
# - If none exist, start with take_001
# ============================================================================
function Get-NextTakeLabel {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ProjectRootPath,

        [Parameter(Mandatory = $true)]
        [string[]]$Gestures,

        [Parameter(Mandatory = $true)]
        [string]$Person,

        [Parameter(Mandatory = $true)]
        [string]$Session
    )

    $takeNumbers = New-Object System.Collections.Generic.List[int]

    foreach ($gestureName in $Gestures) {
        $sessionRoot = Join-Path $ProjectRootPath ("data\\raw\\openpose_json\\{0}\\{1}\\{2}" -f $gestureName, $Person, $Session)
        if (-not (Test-Path -LiteralPath $sessionRoot -PathType Container)) {
            continue
        }

        $takeDirs = Get-ChildItem -LiteralPath $sessionRoot -Directory -ErrorAction SilentlyContinue
        foreach ($takeDir in $takeDirs) {
            if ($takeDir.Name -match '^take_(\d+)$') {
                $takeNumbers.Add([int]$matches[1])
            }
        }
    }

    if ($takeNumbers.Count -eq 0) {
        return "take_001"
    }

    $maxTake = ($takeNumbers | Measure-Object -Maximum).Maximum
    return ("take_{0:D3}" -f ($maxTake + 1))
}

# ============================================================================
# Helper function: run one timed OpenPose capture for a specific gesture.
# This reuses the same launch/readiness/timed-stop pattern as timed helper.
# ============================================================================
function Invoke-TimedOpenPoseCapture {
    param(
        [Parameter(Mandatory = $true)]
        [string]$OpenPoseExePath,

        [Parameter(Mandatory = $true)]
        [string]$OpenPoseWorkingDir,

        [Parameter(Mandatory = $true)]
        [string]$JsonDir,

        [Parameter(Mandatory = $true)]
        [bool]$UseVideo,

        [Parameter(Mandatory = $false)]
        [string]$VideoPath = "",

        [Parameter(Mandatory = $true)]
        [double]$DurationSeconds,

        [Parameter(Mandatory = $true)]
        [int]$CountdownSeconds,

        [Parameter(Mandatory = $true)]
        [int]$PollIntervalMs,

        [Parameter(Mandatory = $true)]
        [int]$StartupTimeoutSeconds
    )

    # Safety check for timed mode:
    # The readiness detector relies on the first new JSON file appearance.
    $existingJson = Get-ChildItem -LiteralPath $JsonDir -Filter "*.json" -File -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($existingJson) {
        throw "JSON output directory is not empty: $JsonDir. Use a new take name or clear this folder before recording."
    }

    Write-Host ""
    Write-Host "Get ready..." -ForegroundColor Cyan
    for ($i = $CountdownSeconds; $i -ge 1; $i--) {
        Write-Host ("{0}..." -f $i)
        Start-Sleep -Seconds 1
    }
    Write-Host "GO" -ForegroundColor Green

    # Build OpenPose arguments as a single string to preserve quoted paths.
    $openPoseArgs = "--number_people_max 1 --tracking 1 --write_json `"$JsonDir`""
    if ($UseVideo) {
        $openPoseArgs += " --write_video `"$VideoPath`""
    }

    $openPoseProcess = $null
    try {
        Write-Host ""
        Write-Host "Launching OpenPose..." -ForegroundColor Cyan
        Write-Host ("OpenPose arguments: {0}" -f $openPoseArgs) -ForegroundColor DarkGray

        $openPoseProcess = Start-Process -FilePath $OpenPoseExePath -ArgumentList $openPoseArgs -WorkingDirectory $OpenPoseWorkingDir -PassThru
        Write-Host ("OpenPose started (PID: {0})." -f $openPoseProcess.Id) -ForegroundColor Green

        Write-Host ("Waiting for capture to start (timeout: {0}s)..." -f $StartupTimeoutSeconds) -ForegroundColor Yellow

        $startupStopwatch = [System.Diagnostics.Stopwatch]::StartNew()
        $captureIsLive = $false

        while ($startupStopwatch.Elapsed.TotalSeconds -lt $StartupTimeoutSeconds) {
            if ($openPoseProcess.HasExited) {
                throw "OpenPose exited before capture started and no JSON output was produced. Arguments used: $openPoseArgs"
            }

            $firstJson = Get-ChildItem -LiteralPath $JsonDir -Filter "*.json" -File -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($firstJson) {
                $captureIsLive = $true
                break
            }

            Start-Sleep -Milliseconds $PollIntervalMs
        }

        if (-not $captureIsLive) {
            Write-Host "Capture startup timeout reached. Stopping OpenPose..." -ForegroundColor Yellow

            if (-not $openPoseProcess.HasExited) {
                $null = $openPoseProcess.CloseMainWindow()
                Start-Sleep -Milliseconds 800
            }
            if (-not $openPoseProcess.HasExited) {
                Stop-Process -Id $openPoseProcess.Id -Force
            }

            throw "Timed out waiting for OpenPose capture output after $StartupTimeoutSeconds seconds."
        }

        Write-Host "Capture is live. Starting timed recording now." -ForegroundColor Green
        Write-Host ("Timed recording in progress for {0} seconds..." -f $DurationSeconds) -ForegroundColor Cyan

        Start-Sleep -Milliseconds ([int]($DurationSeconds * 1000))

        Write-Host "Stopping OpenPose..." -ForegroundColor Yellow

        if (-not $openPoseProcess.HasExited) {
            $null = $openPoseProcess.CloseMainWindow()
            Start-Sleep -Milliseconds 800
        }

        if (-not $openPoseProcess.HasExited) {
            Write-Host "OpenPose still running. Forcing stop..." -ForegroundColor Yellow
            Stop-Process -Id $openPoseProcess.Id -Force
        }

        Start-Sleep -Milliseconds 500
    }
    catch {
        if ($openPoseProcess -and -not $openPoseProcess.HasExited) {
            Stop-Process -Id $openPoseProcess.Id -Force -ErrorAction SilentlyContinue
        }
        throw
    }
}

Write-Host ""
Write-Host "=== OpenPose Gesture Cycle Helper ===" -ForegroundColor Cyan
Write-Host ""

# ============================================================================
# Validate static paths early.
# ============================================================================
if (-not (Test-Path -LiteralPath $OpenPoseRoot -PathType Container)) {
    Write-Error "OpenPose root not found: $OpenPoseRoot"
    exit 1
}

if (-not (Test-Path -LiteralPath $ProjectRoot -PathType Container)) {
    Write-Error "Project root not found: $ProjectRoot"
    exit 1
}

$OpenPoseExe = Join-Path $OpenPoseRoot "bin\\OpenPoseDemo.exe"
if (-not (Test-Path -LiteralPath $OpenPoseExe -PathType Leaf)) {
    Write-Error "OpenPoseDemo.exe not found: $OpenPoseExe"
    exit 1
}

# ============================================================================
# 1) Prompt once for cycle metadata.
# ============================================================================
$person = (Read-Value -Prompt "Person" -Default $DefaultPerson).Trim()
if ([string]::IsNullOrWhiteSpace($person)) {
    $person = $DefaultPerson
}

$session = (Read-Value -Prompt "Session" -Default $DefaultSession).Trim()
if ([string]::IsNullOrWhiteSpace($session)) {
    $session = $DefaultSession
}

$defaultVideoText = if ($DefaultUseVideo) { "y" } else { "n" }
$videoInput = (Read-Value -Prompt "Save video? (y/n)" -Default $defaultVideoText).Trim().ToLowerInvariant()
$useVideo = switch ($videoInput) {
    "" { $DefaultUseVideo }
    "y" { $true }
    "yes" { $true }
    "n" { $false }
    "no" { $false }
    default {
        Write-Warning "Unrecognized input '$videoInput'. Using default '$defaultVideoText'."
        $DefaultUseVideo
    }
}

$durationInput = (Read-Value -Prompt "Recording duration in seconds" -Default "$DefaultDurationSeconds").Trim()
[double]$durationSeconds = 0
if (-not [double]::TryParse($durationInput, [ref]$durationSeconds) -or $durationSeconds -le 0) {
    Write-Error "Duration must be a positive number."
    exit 1
}

# ============================================================================
# 2) Resolve one take label for the whole cycle.
# ============================================================================
$take = Get-NextTakeLabel -ProjectRootPath $ProjectRoot -Gestures $GestureCycle -Person $person -Session $session

Write-Host ""
Write-Host ("This cycle will use {0}" -f $take) -ForegroundColor Yellow
Write-Host ""

# Keep track of successful captures for final summary.
$successfulGestures = New-Object System.Collections.Generic.List[string]

# ============================================================================
# 3) Walk through fixed gesture order.
# ============================================================================
for ($index = 0; $index -lt $GestureCycle.Count; $index++) {
    $gesture = $GestureCycle[$index]
    $position = $index + 1

    $jsonDir = Join-Path $ProjectRoot ("data\\raw\\openpose_json\\{0}\\{1}\\{2}\\{3}" -f $gesture, $person, $session, $take)
    $videoDir = Join-Path $ProjectRoot ("data\\raw\\rgb_video\\{0}\\{1}\\{2}" -f $gesture, $person, $session)
    $videoPath = Join-Path $videoDir ("{0}.avi" -f $take)

    Write-Host "============================================================" -ForegroundColor DarkCyan
    Write-Host ("Gesture      : {0}" -f $gesture) -ForegroundColor Cyan
    Write-Host ("Cycle step   : {0}/{1}" -f $position, $GestureCycle.Count) -ForegroundColor Cyan
    if ($position -lt $GestureCycle.Count) {
        Write-Host ("Next gesture : {0}" -f $GestureCycle[$index + 1]) -ForegroundColor Cyan
    }
    else {
        Write-Host "Next gesture : (this is the last gesture)" -ForegroundColor Cyan
    }
    Write-Host ("Take label   : {0}" -f $take) -ForegroundColor Cyan

    $continueInput = (Read-Host "Press ENTER to record this gesture, or type q to quit the cycle.").Trim().ToLowerInvariant()
    if ($continueInput -eq "q") {
        Write-Host "Cycle stopped by user request." -ForegroundColor Yellow
        break
    }

    # Create directories for this gesture.
    New-Item -ItemType Directory -Path $jsonDir -Force | Out-Null
    if ($useVideo) {
        New-Item -ItemType Directory -Path $videoDir -Force | Out-Null
    }

    # Refuse to continue if JSON folder already contains files.
    $existingJson = Get-ChildItem -LiteralPath $jsonDir -Filter "*.json" -File -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($existingJson) {
        Write-Error "Target JSON directory already has JSON files: $jsonDir"
        Write-Error "Please use a new take label or clear this folder before recording."
        exit 1
    }

    try {
        Invoke-TimedOpenPoseCapture `
            -OpenPoseExePath $OpenPoseExe `
            -OpenPoseWorkingDir $OpenPoseRoot `
            -JsonDir $jsonDir `
            -UseVideo $useVideo `
            -VideoPath $videoPath `
            -DurationSeconds $durationSeconds `
            -CountdownSeconds $DefaultCountdownSeconds `
            -PollIntervalMs $CapturePollIntervalMs `
            -StartupTimeoutSeconds $CaptureStartupTimeoutSeconds

        $successfulGestures.Add($gesture)

        Write-Host ""
        Write-Host ("Success: recorded {0}" -f $gesture) -ForegroundColor Green
        Write-Host ("JSON saved to : {0}" -f $jsonDir)
        if ($useVideo) {
            Write-Host ("Video saved to: {0}" -f $videoPath)
        }
        else {
            Write-Host "Video saving was disabled for this cycle."
        }
    }
    catch {
        Write-Error "Failed while recording '$gesture': $($_.Exception.Message)"
        exit 1
    }
}

# ============================================================================
# 4) Final cycle summary.
# ============================================================================
Write-Host ""
Write-Host "====================== Cycle complete ======================" -ForegroundColor Green
Write-Host ("Take label used: {0}" -f $take) -ForegroundColor Green
Write-Host "Successfully recorded gestures:" -ForegroundColor Green

if ($successfulGestures.Count -eq 0) {
    Write-Host "- (none)"
}
else {
    foreach ($recordedGesture in $successfulGestures) {
        Write-Host ("- {0}" -f $recordedGesture)
    }
}
