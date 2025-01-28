@echo off

REM --------------------------------------------------
REM  run_pipeline.bat
REM  Windows batch script using PowerShell Tee-Object
REM --------------------------------------------------

REM Check if .venv directory exists
IF NOT EXIST ".venv" (
    echo [ERROR] .venv folder not found in the current directory.
    echo Please create a virtual environment named .venv first.
    goto end
)

REM Activate the virtual environment
call .venv\Scripts\activate

REM Use PowerShell to install/upgrade packages and tee output
echo Installing or updating Python requirements (please wait)...

powershell -NoProfile -Command "pip install --upgrade pip | Tee-Object -FilePath pip_install_log.txt"
IF ERRORLEVEL 1 (
    echo [ERROR] Failed to upgrade pip. Check pip_install_log.txt for details.
    goto end
)

powershell -NoProfile -Command "pip install -r requirements.txt | Tee-Object -FilePath pip_install_log.txt"
IF ERRORLEVEL 1 (
    echo [ERROR] Failed to install dependencies. Check pip_install_log.txt for details.
    goto end
)

echo Requirements installed successfully.

echo.
echo --------------------------------------------------
echo Automated RGC Counting - Interactive Prompt
echo --------------------------------------------------

REM Prompt for input directory
set /p input_dir="Enter the folder path that contains the .tif images: "

REM Prompt for output directory
set /p output_dir="Enter the folder path where you want the results saved: "

REM Prompt for approximate diameter
echo.
echo (If you don't know the diameter in pixels, press Enter to let Cellpose auto-estimate.)
set /p diameter_input="Approximate RGC diameter (in pixels), or leave blank: "

if "%diameter_input%"=="" (
    set diameter_arg=
) else (
    set diameter_arg=--diameter %diameter_input%
)

REM Prompt for debug overlay
echo.
set /p debug_choice="Do you want to save debug overlays? (y/n): "
if /i "%debug_choice%"=="y" (
    set debug_arg=--save_debug
) else (
    set debug_arg=
)

REM Prompt for GPU usage
echo.
set /p gpu_choice="Do you want to use GPU for Cellpose? (y/n): "
if /i "%gpu_choice%"=="y" (
    set gpu_arg=--use_gpu
) else (
    set gpu_arg=
)

REM Prompt for applying Gaussian blur
echo.
set /p blur_choice="Apply Gaussian blur for blurry edges? (y/n): "
if /i "%blur_choice%"=="y" (
    set blur_arg=--apply_blur
) else (
    set blur_arg=
)

echo.
echo --------------------------------------------------
echo Now running the main pipeline...
echo (Logging console output to run_pipeline_log.txt)
echo --------------------------------------------------

REM Use PowerShell so we can see live output in the console
REM and also save it to run_pipeline_log.txt
powershell -NoProfile -Command ^
    "python -u main.py ^
        --input_dir '%input_dir%' ^
        --output_dir '%output_dir%' ^
        %diameter_arg% ^
        %debug_arg% ^
        %gpu_arg% ^
        %blur_arg% ^
    | Tee-Object -FilePath run_pipeline_log.txt"

echo.
echo [INFO] All done! Check your output folder for results.
echo [INFO] See run_pipeline_log.txt for the full console log.

:end
pause
