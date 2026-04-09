# OpenPose Recording Helpers (Windows)

These scripts help you record gesture data with **OpenPose** and save the outputs directly into this repository's dataset folders.

- OpenPose is expected to be installed **outside this repo**.
- This repo is used as the **destination** for saved JSON and optional video files.

## Files

- `record_gesture.bat` - edit variables at the top, then run.
- `record_gesture_interactive.bat` - prompts you in the terminal for gesture/person/session/take.

## Variables to edit before each recording

In `record_gesture.bat`, update the `EDIT THESE VALUES` section:

- `OPENPOSE_ROOT` (path to your OpenPose folder)
- `PROJECT_ROOT` (path to this repository)
- `GESTURE`
- `PERSON`
- `SESSION`
- `TAKE`
- `USE_VIDEO` (`1` or `0`)

In `record_gesture_interactive.bat`, update defaults near the top:

- `OPENPOSE_ROOT`
- `PROJECT_ROOT`
- `DEFAULT_PERSON` (default: `luis`)
- `DEFAULT_SESSION` (default: `s01`)

## Example values

Example 1:

- `GESTURE=attack_fire`
- `PERSON=luis`
- `SESSION=s01`
- `TAKE=take_001`

Example 2:

- `GESTURE=defense_water`
- `PERSON=luis`
- `SESSION=s01`
- `TAKE=take_002`

## Output paths created automatically

JSON output:

`%PROJECT_ROOT%\data\raw\openpose_json\%GESTURE%\%PERSON%\%SESSION%\%TAKE%\`

Optional video output (`USE_VIDEO=1`):

`%PROJECT_ROOT%\data\raw\rgb_video\%GESTURE%\%PERSON%\%SESSION%\%TAKE%.avi`

## Suggested recording pattern for one take

For cleaner training examples, one take should include:

1. Neutral pose
2. Gesture
3. Return to neutral

## Example Windows paths (placeholders)

- `OPENPOSE_ROOT=D:\Programs\OpenPose\openpose`
- `PROJECT_ROOT=D:\Documentos\Python Projects\Avatar-Arcade-Game`

## Timed helper (PowerShell)

Use `record_gesture_timed.ps1` when you want a fixed-length capture workflow that is friendly for beginners:

- prompts for gesture/person/session/take
- creates output folders automatically
- gives a short countdown
- launches OpenPose
- records for a fixed duration (default 3 seconds)
- stops OpenPose automatically
- prints where JSON/video were saved

Why fixed-duration capture helps:

- Every take uses the same time budget, which is easier to repeat consistently.
- It speeds up collection because you do not need to stop OpenPose manually each time.
- It keeps raw recordings uniform even before later trimming or model windowing.

Example usage from PowerShell:

```powershell
cd D:\Documentos\Python Projects\Avatar-Arcade-Game\tools
.\record_gesture_timed.ps1
```

You can edit defaults at the top of the script (`$OpenPoseRoot`, `$ProjectRoot`, person/session/video/duration/countdown) to match your machine and preferred settings.

> Note: the raw captured clip can still be longer than the final model input window used downstream. This helper only standardizes capture-time duration.


## Guided full-cycle helper (PowerShell)

Use `record_gesture_cycle.ps1` when you want to capture one full 8-gesture cycle for the same person/session in one guided run.

What it does:

- prompts once for person/session/video/duration
- auto-assigns the next `take_###` label by scanning existing takes across all 8 gestures for that person/session
- walks you through the fixed gesture order below
- waits for ENTER before each capture (or `q` to quit)
- uses the same timed capture behavior as the timed helper (countdown, launch OpenPose, wait for first JSON, timed stop)

Fixed gesture order:

1. `attack_earth`
2. `attack_fire`
3. `attack_water`
4. `attack_air`
5. `defense_earth`
6. `defense_fire`
7. `defense_water`
8. `defense_air`

Example usage from PowerShell:

```powershell
cd D:\Documentos\Python Projects\Avatar-Arcade-Game\tools
.\record_gesture_cycle.ps1
```

This helper is useful when recording one complete cycle with the same `person` and `session` while keeping one consistent auto-assigned take label across all eight gestures.
