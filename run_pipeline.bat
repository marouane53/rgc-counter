@echo off

REM --------------------------------------------------
REM  run_pipeline.bat
REM  Windows batch script for Cell Counting Project
REM --------------------------------------------------

REM Check if .venv directory exists
IF NOT EXIST ".venv" (
    echo [ERROR] .venv folder not found in the current directory.
    echo Please create a virtual environment named .venv first.
    pause
    goto end
)

REM Activate the virtual environment (ensure this is the correct path)
call .venv\Scripts\activate

echo Installing or updating Python requirements (please wait)...

REM Quietly upgrade pip (hide all output, log errors if they occur)
pip install --upgrade pip --quiet > pip_install_log.txt 2>&1
IF ERRORLEVEL 1 (
    echo [ERROR] Failed to upgrade pip. 
    echo Check pip_install_log.txt for details.
    pause
    goto end
)

REM Quietly install dependencies
pip install -r requirements.txt --quiet >> pip_install_log.txt 2>&1
IF ERRORLEVEL 1 (
    echo [ERROR] Failed to install dependencies.
    echo Check pip_install_log.txt for details.
    pause
    goto end
)

echo Requirements installed successfully.

echo.
echo --------------------------------------------------
echo Automated RGC Counting - Interactive Prompt
echo --------------------------------------------------

REM Prompt for input directory (optional)
echo Press Enter to use default "input" folder, or specify a different path:
set /p input_dir="Enter the folder path that contains the .tif images (optional): "

REM Use default if empty
if "%input_dir%"=="" (
    set input_dir=input
    echo Using default input folder: input
)

echo.
echo Press Enter to use default "Outputs" folder, or specify a different path:
set /p output_dir="Enter the folder path where you want the results saved (optional): "

if "%output_dir%"=="" (
    set output_dir=Outputs
    echo Using default output folder: Outputs
)

echo.
echo (If you don't know the diameter in pixels, press Enter to let Cellpose auto-estimate.)
set /p diameter_input="Approximate RGC diameter (in pixels), or leave blank: "

if "%diameter_input%"=="" (
    set diameter_arg=
) else (
    set diameter_arg=--diameter %diameter_input%
)

echo.
set /p debug_choice="Do you want to save debug overlays? (y/n): "
if /I "%debug_choice%"=="y" (
    set debug_arg=--save_debug
) else (
    set debug_arg=
)

echo.
set /p gpu_choice="Do you want to use GPU for Cellpose? (y/n): "
if /I "%gpu_choice%"=="y" (
    set gpu_arg=--use_gpu
) else (
    set gpu_arg=
)

echo.
set /p clahe_choice="Apply CLAHE for contrast enhancement? (y/n): "
if /I "%clahe_choice%"=="y" (
    set clahe_arg=--apply_clahe
) else (
    set clahe_arg=
)

echo.
echo --------------------------------------------------
echo Choose focus mode for each image:
echo (1) No focus bounding (analyze entire image)
echo (2) Manual bounding-box in Napari
echo (3) Automatic tile-based focus detection (improved)
echo --------------------------------------------------
set /p focus_mode_choice="Enter 1, 2, or 3: "

if "%focus_mode_choice%"=="1" goto focus_none
if "%focus_mode_choice%"=="2" goto focus_bbox
if "%focus_mode_choice%"=="3" goto focus_auto
goto focus_invalid

:focus_none
set "focus_mode_arg=--focus_none"
goto focus_done

:focus_bbox
set "focus_mode_arg=--focus_bbox"
goto focus_done

:focus_auto
set "focus_mode_arg=--focus_auto"
goto focus_done

:focus_invalid
echo Invalid choice. Defaulting to (1) No focus bounding.
set "focus_mode_arg=--focus_none"

:focus_done
echo.
echo --------------------------------------------------
echo Now running the main pipeline...
echo --------------------------------------------------

python main.py ^
  --input_dir "%input_dir%" ^
  --output_dir "%output_dir%" ^
  %diameter_arg% ^
  %debug_arg% ^
  %gpu_arg% ^
  %clahe_arg% ^
  %focus_mode_arg%

echo Return code: %ERRORLEVEL%
pause

:end
pause
