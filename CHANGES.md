# Directory Structure & Path Updates

## Summary of Changes

All testing files have been updated to use **relative paths** instead of hardcoded absolute paths. This makes the project portable and ensures scripts work regardless of where the project directory is located.

## New Directory Structure

```
CV_Project/
├── README.md                      # Main project documentation (UPDATED)
├── CHANGES.md                     # This file
├── data/
│   └── stl10_binary/              # STL-10 dataset
├── main/
│   ├── main.ipynb                 # Training notebook
│   ├── colorization_model.pth     # Trained generator
│   └── colorization_discriminator.pth  # Trained discriminator
├── testing/                       # All inference scripts (UPDATED PATHS)
│   ├── inference.py
│   ├── test_inference.py          # UPDATED - Uses relative paths
│   ├── example_usage.py           # UPDATED - Uses relative paths
│   └── README_inference.md        # UPDATED - New location notes
├── test_images/
│   └── migrant.jpg                # Example test image
└── test_results/                  # Output directory
```

## Files Updated

### 1. `/testing/test_inference.py`
**Changes:**
- Use `Path()` for robust relative path resolution
- Model: `../main/colorization_model.pth` (relative)
- Image: `../test_images/migrant.jpg` (relative)
- Output: `../test_results` (relative)
- Import: `from inference import run_inference` (local import)

**Usage:**
```bash
cd testing/
python test_inference.py
```

### 2. `/testing/example_usage.py`
**Changes:**
- Use `Path()` for robust relative path resolution
- Model: `../main/colorization_model.pth` (relative)
- Image: `../test_images/migrant.jpg` (relative)
- Import: `from inference import ColorizationUNet, ...` (local import)

**Usage:**
```bash
cd testing/
python example_usage.py
```

### 3. `/testing/README_inference.md`
**Changes:**
- Added directory structure visualization
- Updated all examples with correct relative paths
- Added `Relative Paths` section explaining the path structure
- All examples now use `../main/` and `../test_images/` paths

### 4. `/README.md` (Main Project)
**Changes:**
- Updated installation to use generic paths (not hardcoded absolute)
- Updated quick test to show: `cd testing/` first
- Updated command line examples with relative paths
- Updated project structure section with new layout
- Added directory descriptions
- Updated programmatic usage example with Path() usage

## Path Resolution

All relative paths are resolved using Python's `pathlib.Path`:

```python
from pathlib import Path

testing_dir = Path(__file__).parent          # testing/ directory
project_root = testing_dir.parent             # CV_Project/ directory

model_path = str(project_root / "main" / "colorization_model.pth")
image_path = str(project_root / "test_images" / "migrant.jpg")
output_dir = str(project_root / "test_results")
```

## Verification

The paths have been tested and work correctly:
- ✅ `test_inference.py` resolves paths correctly
- ✅ `example_usage.py` uses relative paths
- ✅ All README files reference correct paths
- ✅ Scripts work from any directory (no hardcoded absolute paths)

## Running Tests

### Quick Test
```bash
cd CV_Project/testing/
python test_inference.py
```

### Single Image Colorization
```bash
cd CV_Project/testing/
python inference.py --model ../main/colorization_model.pth --images ../test_images/migrant.jpg
```

### Batch Processing
```bash
cd CV_Project/testing/
python inference.py --model ../main/colorization_model.pth --images ../test_images/*.jpg --output ../test_results/batch/
```

## Benefits

1. **Portability**: Project works from any directory
2. **No Hardcoding**: No absolute paths dependent on user's system
3. **Consistency**: All scripts follow the same pattern
4. **Maintainability**: Easier to manage file locations
5. **Collaboration**: Works for other users without path modifications
