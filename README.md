# 3D Thermal Index Toolpath Viewer

This is a Dash-based web application for visualizing 3D thermal index toolpaths.

## Features
- Upload a data file to visualize 3D toolpaths.
- Highlight segments based on a thermal threshold.
- Toggle between cumulative and single-layer views.
- Adjust layer and element sequence interactively.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/thermal-toolpath-inspector.git
   cd thermal-toolpath-inspector
   ```

2. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the app locally:
```bash
python app/inspector.py
```

Open your browser and navigate to `http://127.0.0.1:8050`.

## File Format

The uploaded file should be a whitespace-delimited text file with the following columns:
- `element`: Element index.
- `layer`: Layer index.
- `x`, `y`, `z`: Coordinates.
- `thermal`: Thermal index value.
