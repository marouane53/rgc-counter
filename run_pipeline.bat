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

REM Activate the virtual environment
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

REM Prompt for output directory (optional)
echo.
echo Press Enter to use default "Outputs" folder, or specify a different path:
set /p output_dir="Enter the folder path where you want the results saved (optional): "

REM Use default if empty
if "%output_dir%"=="" (
    set output_dir=Outputs
    echo Using default output folder: Outputs
)

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
set /p blur_choice="Apply Gaussian blur for smoothing edges? (y/n): "
if /i "%blur_choice%"=="y" (
    set blur_arg=--apply_blur
) else (
    set blur_arg=
)

REM Prompt for applying CLAHE
echo.
set /p clahe_choice="Apply CLAHE for contrast enhancement? (y/n): "
if /i "%clahe_choice%"=="y" (
    set clahe_arg=--apply_clahe
) else (
    set clahe_arg=
)

echo.
echo --------------------------------------------------
echo Now running the main pipeline...
echo --------------------------------------------------

REM Run main.py with user-supplied args
python main.py ^
  --input_dir "%input_dir%" ^
  --output_dir "%output_dir%" ^
  %diameter_arg% ^
  %debug_arg% ^
  %gpu_arg% ^
  %blur_arg% ^
  %clahe_arg%

REM If something failed, we keep the window open.
IF ERRORLEVEL 1 (
    echo [ERROR] The Python script exited with an error.
    pause
    goto end
)

echo.
echo [INFO] All done! Check your output folder for results.

:end
pause
