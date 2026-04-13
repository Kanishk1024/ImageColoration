# Colorization Inference Script

This script allows you to run colorization inference on grayscale images using a trained ColorizationUNet model.

## Quick Test

To quickly test the inference script with the provided model and example image:

```bash
python test_inference.py
```

This will colorize the `migrant.jpg` image using the trained `colorization_model.pth` and save results to `test_results/`.

## Usage

### Command Line Interface

```bash
python inference.py --model path/to/model.pth --images image1.jpg image2.png --output results/
```

### Arguments

- `--model`: Path to the trained model checkpoint (required)
- `--images`: One or more paths to input images (required)
- `--output`: Output directory for results (default: 'results')
- `--decode`: Decoding method - 'argmax' (vibrant colors) or 'annealed' (smoother colors) (default: 'argmax')

## Example Usage

```bash
# Colorize a single image with default settings
python inference.py --model checkpoints/model_epoch_10.pth --images test_image.jpg

# Colorize multiple images with smooth decoding
python inference.py --model model.pth --images img1.jpg img2.png img3.jpeg --decode annealed --output colorized_images/

# Colorize images from a directory (using wildcard)
python inference.py --model model.pth --images images/*.jpg
```

## Output

For each input image, the script generates:

1. `{basename}_colorized.png` - The colorized version of the input image
2. `{basename}_grayscale.png` - Grayscale version of the input (for reference)
3. `{basename}_comparison.png` - Side-by-side comparison plot

## Requirements

The script requires the same dependencies as the training notebook:

- PyTorch
- NumPy
- Matplotlib
- scikit-image
- scikit-learn
- Pillow

## Model Format

The script expects a PyTorch model checkpoint. The checkpoint can be either:
- A state_dict dictionary (saved with `torch.save(model.state_dict(), path)`)
- A full checkpoint dictionary containing 'model_state_dict' key (saved during training)

## Image Formats

Supported input formats: JPG, PNG, JPEG, and other formats supported by scikit-image.

The script automatically:
- Converts images to RGB if they're grayscale or have alpha channels
- Resizes images to 128x128 for processing
- Converts back to original size for output