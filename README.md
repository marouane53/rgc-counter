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
├── data/              # Place your input images here
├── output/           # Results will be saved here
├── src/
│   ├── cell_segmentation.py    # Cell detection using Cellpose
│   ├── postprocessing.py       # Post-processing of results
│   ├── analysis.py            # Analysis functions
│   ├── visualize.py           # Visualization tools
│   ├── config.py             # Configuration settings
│   └── utils.py              # Utility functions
├── main.py           # Main script for single image processing
└── run_pipeline.bat  # Batch script for processing multiple images
```

## Usage

### Single Image Processing

To process a single image:

```bash
python main.py --input path/to/image.tif --output path/to/output
```

### Batch Processing

1. Place your images in the `data/` folder
2. Configure settings in `src/config.py` if needed
3. Run the batch script:
   - Double-click `run_pipeline.bat`, or
   - Open command prompt and run:
```bash
run_pipeline.bat
```

The batch script will:
- Process all images in the `data/` folder
- Save results in the `output/` folder
- Generate analysis reports and visualizations

## Output Structure

For each processed image, you'll find:
- `masks/`: Segmentation masks
- `visualization/`: Visual results and overlays
- `analysis/`: CSV files with cell counts and metrics
- `report.pdf`: Summary report with key findings

## Configuration

Edit `src/config.py` to customize:
- Cell detection parameters
- Analysis settings
- Output formats
- Visualization options

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
- Contributors and maintainers

## Support

For support, please open an issue in the GitHub repository or contact the maintainers.
