#!/usr/bin/env python3
"""
Example script showing how to use the colorization inference functionality programmatically.
"""

import torch
from pathlib import Path
from inference import ColorizationUNet, build_synthetic_gamut, preprocess_image, postprocess_image, argmax_decode

def colorize_single_image(model_path, image_path):
    """
    Example function showing how to colorize a single image programmatically.
    """
    # Build the color gamut
    q_ab, _ = build_synthetic_gamut()

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ColorizationUNet(num_bins=313).to(device)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    model.eval()

    # Preprocess the image
    l_input, lab_original = preprocess_image(image_path, image_size=128)
    original_size = lab_original.shape[:2]

    # Run inference
    with torch.no_grad():
        logits = model(l_input)
        ab_pred = argmax_decode(logits, q_ab)

    # Postprocess to get RGB image
    colorized_rgb = postprocess_image(l_input, ab_pred, original_size)

    return colorized_rgb

def batch_colorize_images(model_path, image_paths):
    """
    Example function showing how to colorize multiple images in batch.
    """
    # Build the color gamut
    q_ab, _ = build_synthetic_gamut()

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ColorizationUNet(num_bins=313).to(device)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    model.eval()

    results = {}

    for image_path in image_paths:
        # Preprocess the image
        l_input, lab_original = preprocess_image(image_path, image_size=128)
        original_size = lab_original.shape[:2]

        # Run inference
        with torch.no_grad():
            logits = model(l_input)
            ab_pred = argmax_decode(logits, q_ab)

        # Postprocess to get RGB image
        colorized_rgb = postprocess_image(l_input, ab_pred, original_size)

        results[image_path] = colorized_rgb

    return results

if __name__ == "__main__":
    # Example usage with relative paths
    testing_dir = Path(__file__).parent
    project_root = testing_dir.parent

    model_path = str(project_root / "main" / "colorization_model.pth")
    image_path = str(project_root / "test_images" / "migrant.jpg")

    # Colorize single image
    colorized = colorize_single_image(model_path, image_path)
    print(f"Colorized image shape: {colorized.shape}")

    # You can now save or display the colorized image
    # from PIL import Image
    # img = Image.fromarray((colorized * 255).astype('uint8'))
    # img.save("colorized_output.png")