@echo off
chcp 65001 >nul
echo ========================================
echo Step2 Batch Processing - Remaining Videos
echo ========================================
echo.

REM Process each action
for %%A in (pushup squat situp jump_rope long_jump pullup) do (
    echo Processing %%A...
    .venv\Scripts\python.exe 0_preprocess_videos.py --action %%A --model yolov8n-pose.pt
    echo.
)

echo ========================================
echo All actions processed!
echo ========================================
pause
