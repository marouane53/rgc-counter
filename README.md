# RGC Counter

A Python-based tool for automated Retinal Ganglion Cell (RGC) counting and analysis using deep learning.

## Description

This project provides an automated pipeline for:
- Cell segmentation using Cellpose
- Intelligent focus region detection
- Post-processing of segmentation results
- Analysis and visualization of cell counts
- Interactive batch processing with customizable parameters

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster processing)
- Windows 10 or higher

## Installation

1. Clone the repository:
```bash
git clone https://github.com/marouane53/rgc-counter.git
cd rgc-counter
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Project Structure

```
rgc-counter/
├── input/             # Place your input .tif images here (not tracked by git)
├── Outputs/           # Results will be saved here (not tracked by git)
├── src/
│   ├── cell_segmentation.py    # Cell detection using Cellpose
│   ├── focus_detection.py      # Focus region detection
│   ├── postprocessing.py       # Post-processing of results
│   ├── analysis.py            # Analysis functions
│   ├── visualize.py           # Visualization tools
│   ├── config.py             # Configuration settings
│   └── utils.py              # Utility functions
├── main.py           # Main script for image processing
├── config.yaml       # Configuration file for pipeline settings
└── run_pipeline.bat  # Interactive batch script with parameter customization
```

## Usage

### Interactive Batch Processing

The easiest way to use the pipeline is through the interactive batch script:

1. Place your .tif images in the `input/` folder
2. Double-click `run_pipeline.bat` or run it from the command prompt
3. Follow the interactive prompts to configure:
   - Input/output directories
   - Cell diameter (or let Cellpose auto-estimate)
   - Debug overlay options
   - GPU usage
   - CLAHE contrast enhancement
   - Focus detection mode

### Focus Detection Modes

The pipeline offers three focus detection modes:

1. **No Focus Bounding**: Analyzes the entire image without focus detection
2. **Manual Bounding-Box**: Opens Napari for manual selection of focus regions
3. **Automatic Tile-based**: Uses improved automatic focus detection algorithm

### Command Line Usage

For direct command line usage:

```bash
python main.py --input_dir "input" --output_dir "Outputs" [OPTIONS]
```

Available options:
- `--diameter`: Cell diameter in pixels (optional)
- `--save_debug`: Save debug overlays
- `--use_gpu`: Enable GPU acceleration
- `--apply_clahe`: Apply CLAHE contrast enhancement
- `--focus_none`: Analyze entire image
- `--focus_bbox`: Enable manual focus selection
- `--focus_auto`: Use automatic focus detection

## Output Structure

For each processed image, you'll find in the `Outputs/` directory:
- Segmentation masks
- Focus region maps (if focus detection is enabled)
- Visual results and overlays (if debug mode is enabled)
- CSV files with cell counts and metrics
- The directory structure mirrors your input directory structure

## Configuration

Edit `config.yaml` to customize:
- Cell detection parameters
- Focus detection settings
- Analysis thresholds
- Visualization options
- Output formats

## Performance Tips

1. **GPU Acceleration**: Enable GPU support for significantly faster processing
2. **Focus Detection**: 
   - Use automatic mode for batch processing
   - Manual mode provides highest accuracy but requires user interaction
3. **CLAHE Enhancement**: Enable for images with poor contrast
4. **Cell Diameter**: 
   - Specify if known for better accuracy
   - Leave blank for automatic estimation

## Git Integration

The project is set up to:
- Track all source code and configuration files
- Ignore input images and output results
- Maintain the `input/` and `Outputs/` directory structure
- Prevent accidental commits of large data files

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Cellpose](https://github.com/mouseland/cellpose) for cell segmentation
- [Napari](https://napari.org/) for interactive visualization

## Support

For support, please open an issue in the GitHub repository or contact the maintainers.
