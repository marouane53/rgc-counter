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
set /p backend_choice="Segmentation backend (cellpose/stardist/sam) [default cellpose]: "
if "%backend_choice%"=="" (
    set backend_arg=
) else (
    set backend_arg=--backend %backend_choice%
)

echo.
set /p tta_choice="Enable TTA? (y/n): "
if /I "%tta_choice%"=="y" (
    set tta_arg=--tta
) else (
    set tta_arg=
)

echo.
set /p spatial_choice="Compute spatial stats (NNRI/VDRI/Ripley)? (y/n): "
if /I "%spatial_choice%"=="y" (
    set spatial_arg=--spatial_stats
) else (
    set spatial_arg=
)

echo.
set /p zarr_choice="Save OME-Zarr (image + labels)? (y/n): "
if /I "%zarr_choice%"=="y" (
    set zarr_arg=--save_ome_zarr
) else (
    set zarr_arg=
)

echo.
set /p report_choice="Write HTML report? (y/n): "
if /I "%report_choice%"=="y" (
    set report_arg=--write_html_report
) else (
    set report_arg=
)

echo.
echo (Focus) 1 None  2 BBox  3 Legacy Auto  4 QC Multi-metric
set /p focus_choice="Enter 1-4: "
if "%focus_choice%"=="2" set "focus_mode_arg=--focus_bbox"
if "%focus_choice%"=="3" set "focus_mode_arg=--focus_auto"
if "%focus_choice%"=="4" set "focus_mode_arg=--focus_qc"
if "%focus_choice%"=="1" set "focus_mode_arg=--focus_none"

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
  %focus_mode_arg% ^
  %backend_arg% ^
  %tta_arg% ^
  %spatial_arg% ^
  %zarr_arg% ^
  %report_arg%

echo Return code: %ERRORLEVEL%
pause

:end
pause
