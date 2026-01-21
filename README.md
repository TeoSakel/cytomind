# CytoMind

**CytoMind** is a backend framework for automating and streamlining flow cytometry analysis pipelines. It provides a structured, reproducible approach to processing FCS files with built-in quality control, interactive revision capabilities, and comprehensive data management.

## Features

- **Automated Pipeline Execution**: Run standardized flow cytometry analysis steps with automatic quality control
- **Interactive Revision System**: Fine-tune compensation matrices and other parameters through an iterative revision workflow
- **Quality Control Integration**: Built-in QC evaluators for each pipeline step with detailed metrics and flags
- **Flexible Data Management**: Project-based repository structure with version-controlled analysis steps
- **Compensation Management**: Automatic detection, application, and refinement of spillover compensation matrices
- **Multi-Layer Data Storage**: Support for multiple data layers (raw, compensated, transformed) using AnnData format
- **Batch Processing**: Group samples into batches for consistent processing and analysis
- **Extensible Architecture**: Registry-based plugin system for steps, QC evaluators, and revision handlers

## Pipeline Steps

CytoMind currently supports the following analysis steps:

- **add_samples**: Parse FCS files, extract metadata, build channel panels, and register samples
- **load_fcs**: Load FCS data into AnnData format for analysis
- **compensate**: Apply spillover compensation to fluorescence channels
- **add_layer**: Create new data layers with custom dimension definitions
- **add_dimensions**: Add dimensions to existing data layers
- **transform**: Apply mathematical transformations (logicle, asinh, etc.) to channels

## Installation

### From Source

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/cytomind.git
   cd cytomind
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -e .
   ```

   This will install CytoMind in editable mode along with its dependencies:
   - `flowkit` (>=1.3.0): FCS file parsing and flow cytometry utilities
   - `scanpy` (>=1.11.5): AnnData file operations and single-cell analysis
   - `plotly` (>=6.5.2): Interactive visualizations

### Development Installation

For development, you may want to install additional tools:

```bash
pip install -e ".[dev]"  # install jupyter and other tools used in development
```

## Quick Start

```python
from pathlib import Path
from cytomind import InteractivePipeline

# Create a new project
pipeline = InteractivePipeline(Path("my_project"))

# Add FCS samples
samples = {
    "sample_001": "path/to/sample_001.fcs",
    "sample_002": "path/to/sample_002.fcs",
}
pipeline.add_samples(samples)

# Load FCS data
pipeline.load_fcs()

# Apply compensation
comp_map = {
    "sample_001": "comp_abc123",
    "sample_002": "comp_abc123",
}
step = pipeline.compensate_samples(comp_map)

# Review QC results
review = pipeline.review_step(step.id)
print(review.qc_summary)

# Start interactive revision (if needed)
handler = pipeline.start_revision(
    step.id,
    input_spec={"sample_ids": ["sample_001", "sample_002"]}
)

# Apply refinements
qc_summary = pipeline.apply_revision(
    handler.session.id,
    user_input={"adjustments": {...}}
)

# Commit changes
new_step = pipeline.commit_revision(handler)
```

## Project Structure

A CytoMind project follows this directory structure:

```
my_project/
├── project.json              # Project metadata and configuration
├── dimensions.json           # Channel dimensions and transformations
├── transformations.json      # Transformation definitions
├── samples/                  # Per-sample data storage
│   ├── sample_001/
│   │   ├── sample.json      # Sample metadata
│   │   ├── raw.h5ad         # Raw FCS data
│   │   └── comp.h5ad        # Compensated data
│   └── sample_002/
│       └── ...
├── compensations/            # Compensation matrix catalog
│   ├── catalog.json
│   └── matrices/
│       ├── comp_abc123.csv
│       └── ...
├── batches/                  # Batch processing results
│   └── ...
└── steps/                    # Pipeline execution history
    ├── step_0001/
    │   └── step_0001_add_samples.json
    ├── step_0002/
    │   └── step_0002_compensate.json
    └── ...
```

## Documentation

- **Tutorial Notebook**: See [notebooks/compensation_tutorial.ipynb](notebooks/compensation_tutorial.ipynb) for a comprehensive walkthrough of the compensation workflow
- **API Documentation**: Coming soon

## Requirements

- Python >= 3.10
- FCS files in Flow Cytometry Standard format (3.0 or later)
- Consistent channel panels across samples (or provide channel mapping)

## License

MIT License

Copyright (c) 2026 Teo Sakel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Authors

- Teo Sakel (teo@intelligentbiodata.com)

## Acknowledgments

Built with:
- [FlowKit](https://github.com/whitman537/flowkit) for FCS file parsing
- [Scanpy](https://scanpy.readthedocs.io/) for AnnData infrastructure
- [Plotly](https://plotly.com/) for interactive visualizations
