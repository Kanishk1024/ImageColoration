#!/usr/bin/env python3
"""
Simple test script to demonstrate the colorization inference.
Run this to test the inference script with the available model and image.
"""

import os
import sys
from pathlib import Path

def test_inference():
    """Test the inference script with available model and image."""

    # Get the testing directory path
    testing_dir = Path(__file__).parent
    project_root = testing_dir.parent

    # Define relative paths
    model_path = str(project_root / "main" / "colorization_model.pth")
    image_path = str(project_root / "test_images" / "migrant.jpg")
    output_dir = str(project_root / "test_results")

    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return False

    if not os.path.exists(image_path):
        print(f"Error: Test image '{image_path}' not found.")
        return False

    # Run inference
    print("Testing inference script...")
    print(f"Model: {model_path}")
    print(f"Image: {image_path}")
    print(f"Output directory: {output_dir}")

    # Import and run inference
    try:
        from inference import run_inference
        run_inference(model_path, [image_path], output_dir, use_argmax=True)
        print("Inference completed successfully!")
        print(f"Check the '{output_dir}' directory for results.")
        return True
    except Exception as e:
        print(f"Error during inference: {e}")
        return False

if __name__ == "__main__":
    success = test_inference()
    sys.exit(0 if success else 1)