# AI-Based Image Colorization System

## 🎨 Overview

This project implements an intelligent image colorization system that automatically converts grayscale images into realistic, color versions using deep learning. Unlike simple colorization approaches, this system explicitly models **color ambiguity and uncertainty**, acknowledging that multiple valid colorizations can exist for the same grayscale image.

The system treats colorization as a pixel-wise prediction problem, leveraging the Lab color space to separate luminance (L channel) from chrominance (a and b channels), enabling more stable and semantically meaningful learning.

## 🎯 Key Features

- **GAN-Based Training**: Adversarial training with patch-based discriminator for realistic colorization
- **U-Net Generator**: Robust encoder-decoder with skip connections and global context
- **Lab Color Space Processing**: Operates in Lab space for better color representation and learning stability
- **Uncertainty Awareness**: MC Dropout enables uncertainty quantification in color predictions
- **Argmax Decoding**: Produces vibrant, saturated colors by selecting most confident color bins
- **Global Context Module**: Captures semantic information at the bottleneck to improve color coherence
- **Combined Loss Functions**: Cross-entropy + L1 reconstruction + adversarial loss for optimal results
- **Comprehensive Output**: Generates colorized images, grayscale versions, and side-by-side comparisons

## 🏗️ Technical Approach

### GAN-Based Adversarial Training

The project uses a **Conditional GAN (cGAN)** framework combining:

**Generator (U-Net Backbone):**
- Encoder-Decoder structure with skip connections
- Encoder: Progressive downsampling (64 → 128 → 256 → 512 channels)
- Bottleneck: Deep feature extraction with MC Dropout for uncertainty
- Decoder: Progressive upsampling with skip connections
- Output: 313 quantized color bin predictions

**Discriminator (Patch-Based):**
- Evaluates 70×70 patches of colorization
- Patch-based discrimination catches local artifacts
- Ensures realistic color transitions
- Uses LeakyReLU and batch normalization

**Combined Training Strategy:**
1. **Cross-Entropy Loss**: Preserves color ambiguity awareness
2. **L1 Reconstruction Loss**: Encourages pixel-accurate color matching (weight: 100)
3. **Adversarial Loss**: Discriminator enforces realistic colorization (weight: 1.0)
4. **LSGAN Formulation**: Ensures stable, convergent training

### Advanced Components

1. **Global Context Module**
   - Extracts global semantic priors from bottleneck
   - Uses adaptive average pooling to compress spatial information
   - Provides semantic guidance for color coherence

2. **Epistemic Uncertainty**
   - MC Dropout at bottleneck and decoder layers
   - Enables multiple forward passes to capture color ambiguity
   - Acknowledges that multiple valid colorizations exist

3. **Color Quantization**
   - Predicts from 313 quantized CIE ab color bins
   - Simulates natural color distribution
   - Reduces complexity vs. continuous prediction

### Color Space: Lab

- **Input**: L channel (luminance) normalized to [-1, 1]
- **Output**: Predicted a, b channels (chrominance)
- **Benefits**:
  - Perceptually uniform representation
  - Separates luminance from color information
  - Enables stable training with better convergence

## 📋 Requirements

- Python 3.7+
- PyTorch (GPU recommended, CPU supported)
- NumPy
- Matplotlib
- scikit-image
- scikit-learn
- Pillow

## 🚀 Installation & Setup

### 1. Clone or navigate to the project directory
```bash
cd /path/to/CV_Project
```

### 2. Install dependencies
```bash
pip install torch torchvision numpy matplotlib scikit-image scikit-learn pillow
```

### 3. Verify CUDA (optional, for GPU acceleration)
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## 🎓 Training the Model

The full GAN-based training pipeline is in `main.ipynb`:

```bash
# Open and run the Jupyter notebook
jupyter notebook main.ipynb
```

**Training Overview:**
1. Loads STL-10 dataset (5000 training images)
2. Converts images to Lab color space
3. Defines generator (U-Net) and patch-based discriminator
4. Runs adversarial training for 15 epochs:
   - Generator: Produces colorizations and fools discriminator
   - Discriminator: Distinguishes real from generated colors
5. Saves trained `colorization_model.pth` (generator) and `colorization_discriminator.pth` (discriminator)

**Training Configuration:**
- Batch size: 8
- Learning rate: 2e-4
- Epochs: 15
- Loss weights: CE + 100×L1 + 1×Adversarial
- Optimizer: Adam with betas (0.5, 0.999)

The trained generator from `main.ipynb` is what `inference.py` uses for colorization.

## 💻 Usage

### Quick Test

Run a quick test with the provided model and example image:

```bash
cd testing/
python test_inference.py
```

This will:
- Load the trained model from `../main/colorization_model.pth`
- Colorize the image from `../test_images/migrant.jpg`
- Save results to `../test_results/`

### Command Line Inference

Colorize images from the command line:

```bash
cd testing/
python inference.py --model ../main/colorization_model.pth --images ../test_images/migrant.jpg --output ../test_results/
```

#### Command Line Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--model` | str | ✓ | - | Path to trained model checkpoint (.pth file) |
| `--images` | str(s) | ✓ | - | One or more image paths (space-separated) |
| `--output` | str | ✗ | `results` | Output directory for colorized images and comparisons |
| `--decode` | str | ✗ | `argmax` | Decoding method: `argmax` (vibrant) or `annealed` (smooth) |

### Command Line Examples

#### Single Image with Default Settings
```bash
cd testing/
python inference.py --model ../main/colorization_model.pth --images ../test_images/your_image.jpg
```

#### Multiple Images with Smooth Decoding
```bash
cd testing/
python inference.py --model ../main/colorization_model.pth --images ../test_images/*.jpg --decode annealed --output ../test_results/batch_results/
```

#### Batch Processing with Wildcard
```bash
cd testing/
python inference.py --model ../main/colorization_model.pth --images ../test_images/*.jpg --output ../test_results/batch_results/
```

### Programmatic Usage

Use the colorization functions directly in your Python code:

```python
from pathlib import Path
from inference import ColorizationUNet, build_synthetic_gamut, preprocess_image, postprocess_image, argmax_decode
import torch

# Get relative paths
project_root = Path(__file__).parent.parent  # Adjust based on your script location
model_path = str(project_root / "main" / "colorization_model.pth")
image_path = str(project_root / "test_images" / "migrant.jpg")

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ColorizationUNet(num_bins=313).to(device)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Colorize a single image
q_ab, _ = build_synthetic_gamut()
l_input, lab_original = preprocess_image(image_path, image_size=128)
original_size = lab_original.shape[:2]

with torch.no_grad():
    logits = model(l_input)
    ab_pred = argmax_decode(logits, q_ab)

colorized_rgb = postprocess_image(l_input, ab_pred, original_size)
```

See `testing/example_usage.py` for complete programmatic examples.

## 📁 Project Structure

```
CV_Project/
├── README.md                      # Main project documentation
├── data/
│   └── stl10_binary/              # STL-10 dataset (used for training)
│       ├── class_names.txt
│       └── fold_indices.txt
├── main/                          # Training & model storage
│   ├── main.ipynb                 # Training notebook (GAN-based adversarial training)
│   ├── colorization_model.pth     # Trained generator checkpoint
│   └── colorization_discriminator.pth # Trained discriminator checkpoint
├── testing/                       # Inference scripts
│   ├── inference.py               # Main inference script (generator only)
│   ├── test_inference.py          # Quick test script
│   ├── example_usage.py           # Programmatic usage examples
│   └── README_inference.md        # Inference-specific documentation
├── test_images/                   # Test images
│   └── migrant.jpg                # Example test image
└── test_results/                  # Output directory for colorization results
    └── (generated outputs)
```

### Directory Descriptions

- **main/**: Contains training artifacts and models
  - `main.ipynb`: Complete GAN-based training pipeline with generator and discriminator
  - `colorization_model.pth`: Pre-trained generator (U-Net for colorization)
  - `colorization_discriminator.pth`: Pre-trained discriminator (for reference)

- **testing/**: All inference and testing scripts
  - Uses relative paths to access model and images
  - Import paths are simplified for local module usage

- **test_images/**: Sample images for testing
  - `migrant.jpg`: Example test image for quick validation

- **test_results/**: Generated colorization outputs
  - Created automatically when running inference

## 📊 Output Files

For each processed image, the system generates three output files:

1. **`{basename}_colorized.png`** - The AI-colorized version of the input image
2. **`{basename}_grayscale.png`** - Grayscale reference (for comparison)
3. **`{basename}_comparison.png`** - Side-by-side visualization with:
   - Input grayscale image
   - Colorized output (model prediction)
   - Ground truth (original color image, if available)

## 🎨 Decoding Methods

### Argmax Decoding (Default - Recommended)
- Selects the single most confident color bin per pixel
- Produces **vibrant, saturated colors**
- **Works synergistically with GAN discriminator** to enforce realism
- Recommended for GAN-trained models
- Usage: `--decode argmax`

### Annealed Mean Decoding
- Averages across color bins weighted by softmax probabilities
- Produces **smooth, potentially desaturated colors**
- Better for when you want perceptually pleasing but less saturated results
- Usage: `--decode annealed`

## 🔧 Model Architecture Details

### ColorizationUNet Components

**Encoder Path:**
- Conv Block 1: 1 → 64 channels
- Conv Block 2: 64 → 128 channels  
- Conv Block 3: 128 → 256 channels

**Bottleneck:**
- Conv Block: 256 → 512 channels
- MC Dropout (p=0.5) for uncertainty
- Global Context Module for semantic features

**Decoder Path:**
- Upsampling with skip connections
- Conv Block 1: 512+256 → 256 channels
- Conv Block 2: 256+128 → 128 channels
- Conv Block 3: 128+64 → 64 channels
- Final Conv: 64 → 313 (color bins)

### Color Gamut

The system uses **313 quantized CIE ab color bins**:
- Grid range: -110 to 110 in both a and b dimensions
- Step size: 10
- Circular mask to simulate natural color distribution
- Enables classification-based color prediction

## 🖼️ Supported Image Formats

**Input Formats:** JPG, PNG, JPEG, and other formats supported by scikit-image

**Automatic Preprocessing:**
- Converts grayscale images to RGB
- Removes alpha channels from RGBA images
- Resizes to 128×128 for model input
- Resizes back to original dimensions in output

## 📈 Performance Considerations

- **GPU Acceleration**: Strongly recommended for faster inference
- **Batch Size**: Currently processes one image at a time (can be modified for batch processing)
- **Memory**: ~500MB VRAM for single image processing
- **Speed**: ~0.5-2 seconds per image (varies by GPU)

### GAN Training Benefits

The GAN-based training approach provides several advantages:
- **Vibrant Colors**: Discriminator rewards colorful, realistic outputs
- **Better Texture Coherence**: Patch-based discrimination catches local inconsistencies
- **Reduced Blur**: Argmax decoding + adversarial loss eliminates averaging artifacts
- **Stable Training**: LSGAN formulation ensures convergent training without mode collapse

## 🧪 Example Workflow

```bash
# 1. Test the setup with a quick inference
python test_inference.py

# 2. Colorize your own image with vibrant colors
python inference.py --model colorization_model.pth --images my_photo.jpg --output my_results/

# 3. Try smooth decoding for different aesthetics
python inference.py --model colorization_model.pth --images my_photo.jpg --decode annealed --output smooth_results/

# 4. Batch process multiple images
python inference.py --model colorization_model.pth --images photos/*.jpg --output batch_colorized/
```

## 📝 Model Checkpoint Format

The system supports two checkpoint formats:

1. **State Dict Only**
   ```python
   torch.save(model.state_dict(), "checkpoint.pth")
   ```

2. **Full Checkpoint Dictionary**
   ```python
   torch.save({
       'model_state_dict': model.state_dict(),
       'epoch': epoch,
       'optimizer_state': optimizer.state_dict()
   }, "checkpoint.pth")
   ```

Both formats are automatically detected and handled.

## 🎓 Learning Approach

This project demonstrates:

1. **Color Space Theory**: Why Lab space is superior to RGB for colorization tasks
2. **Uncertainty in CV**: Using MC Dropout to model color ambiguity
3. **Architecture Design**: U-Net with skip connections and global context
4. **Practical Deep Learning**: Training, inference, and deployment of vision models
5. **Modern Techniques**: Attention-like mechanisms via global context modules

## ⚠️ Limitations & Known Issues

1. **Color Ambiguity**: Multiple valid colorizations exist (e.g., clothing, objects)
2. **Desaturation Bias**: Model may produce slightly desaturated colors
3. **Input Size**: Currently designed for 128×128 processing (resizes to this)
4. **Black & White Objects**: May struggle with inherently grayscale objects
5. **Fine Details**: Small, intricate details may lose color precision during resizing

## 🔮 Future Improvements

- [ ] Multi-scale processing for better detail preservation
- [ ] Uncertainty visualization and confidence maps
- [ ] Ensemble methods for multiple plausible colorizations
- [ ] Fine-tuning on domain-specific datasets
- [ ] Real-time video colorization support
- [ ] Interactive user-guided colorization
- [ ] GAN-based training for perceptually better results

## 📚 References

- Original Colorization Paper: Zhang et al., "Colorful Image Colorization" (ECCV 2016)
- U-Net Architecture: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (MICCAI 2015)
- Lab Color Space: Hunter, Richard Sewall. "Photoelectric color-difference meter" (1948)

## 📧 Questions or Issues?

For questions about usage, refer to `README_inference.md` for detailed inference documentation.

## 📄 License

This project is provided as-is for educational and research purposes.

---

**Last Updated:** 2026-04-13
