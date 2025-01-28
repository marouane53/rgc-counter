# RGC Counter

A Python-based tool for automated Retinal Ganglion Cell (RGC) counting and analysis using deep learning.

## Description

This project provides an automated pipeline for:
- Cell segmentation using Cellpose
- Post-processing of segmentation results
- Analysis and visualization of cell counts
- Batch processing capabilities for multiple images

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
├── input/             # Place your input images here (not tracked by git)
├── Outputs/           # Results will be saved here (not tracked by git)
├── src/
│   ├── cell_segmentation.py    # Cell detection using Cellpose
│   ├── postprocessing.py       # Post-processing of results
│   ├── analysis.py            # Analysis functions
│   ├── visualize.py           # Visualization tools
│   ├── config.py             # Configuration settings
│   └── utils.py              # Utility functions
├── main.py           # Main script for single image processing
├── config.yaml       # Configuration file for pipeline settings
└── run_pipeline.bat  # Batch script for processing multiple images
```

## Usage

### Single Image Processing

To process a single image:

```bash
python main.py --input_dir input --output_dir Outputs
```

Additional options:
```bash
python main.py --help
```

### Batch Processing

1. Place your images in the `input/` folder (they won't be tracked by git)
2. Configure settings in `config.yaml` if needed
3. Run the batch script:
   - Double-click `run_pipeline.bat`, or
   - Open command prompt and run:
```bash
run_pipeline.bat
```

The pipeline will:
- Process all images in the `input/` folder and its subdirectories
- Automatically create corresponding output directories
- Save results in the `Outputs/` folder
- Generate analysis reports and visualizations

## Output Structure

For each processed image, you'll find in the `Outputs/` directory:
- Segmentation masks
- Visual results and overlays
- CSV files with cell counts and metrics
- The directory structure will mirror your input directory structure

## Configuration

Edit `config.yaml` to customize:
- Cell detection parameters
- Analysis settings
- Output formats
- Visualization options

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

## Support

For support, please open an issue in the GitHub repository or contact the maintainers.
