from pathlib import Path
from PIL import Image, ImageOps, ImageFilter
import math, argparse, sys
from tqdm import tqdm
import os
import numpy as np

def get_dominant_background_color(image, sample_size=10):
    """
    Get the dominant background color by sampling corners and edges
    """
    width, height = image.size
    corners = [
        image.getpixel((0, 0)),
        image.getpixel((width-1, 0)),
        image.getpixel((0, height-1)),
        image.getpixel((width-1, height-1))
    ]
    
    # Sample edges
    edge_samples = []
    for i in range(0, width, max(1, width//sample_size)):
        edge_samples.append(image.getpixel((i, 0)))  # Top edge
        edge_samples.append(image.getpixel((i, height-1)))  # Bottom edge
    for i in range(0, height, max(1, height//sample_size)):
        edge_samples.append(image.getpixel((0, i)))  # Left edge
        edge_samples.append(image.getpixel((width-1, i)))  # Right edge
    
    # Combine all samples
    all_samples = corners + edge_samples
    
    # Find most common color (simple mode)
    from collections import Counter
    color_count = Counter(all_samples)
    return color_count.most_common(1)[0][0]

def smart_crop_center(image, target_size):
    """
    Smart center crop that tries to preserve important content
    """
    width, height = image.size
    
    # If image is already square-ish, use regular center crop
    aspect_ratio = width / height
    if 0.8 <= aspect_ratio <= 1.2:
        return ImageOps.fit(image, (target_size, target_size), Image.Resampling.LANCZOS)
    
    # For very wide images, try to detect if there's a central subject
    if aspect_ratio > 1.5:
        # Portrait crop from landscape - focus on center
        crop_width = min(width, int(height * 1.1))  # Slightly wider than square
        left = (width - crop_width) // 2
        image = image.crop((left, 0, left + crop_width, height))
    elif aspect_ratio < 0.7:
        # Landscape crop from portrait - focus on upper center
        crop_height = min(height, int(width * 1.1))  # Slightly taller than square
        top = max(0, (height - crop_height) // 3)  # Bias towards upper third
        image = image.crop((0, top, width, top + crop_height))
    
    # Final resize to exact target size
    return ImageOps.fit(image, (target_size, target_size), Image.Resampling.LANCZOS)

def create_thumbnail_no_background(image, thumb_px, method="crop"):
    """
    Create thumbnail without white background using different methods
    
    Args:
        image: PIL Image object
        thumb_px: Target thumbnail size
        method: "crop", "fit_transparent", "fit_blur", or "fit_extend"
    """
    original_width, original_height = image.size
    aspect_ratio = original_width / original_height
    
    if method == "crop":
        # Smart center crop - no background, fills entire thumbnail
        return smart_crop_center(image, thumb_px)
    
    elif method == "fit_transparent":
        # Fit image maintaining aspect ratio with transparent background
        # Note: This creates RGBA image, which may not work with all sprite viewers
        image.thumbnail((thumb_px, thumb_px), Image.Resampling.LANCZOS)
        square = Image.new("RGBA", (thumb_px, thumb_px), (0, 0, 0, 0))  # Transparent
        offset = ((thumb_px - image.width) // 2, (thumb_px - image.height) // 2)
        square.paste(image, offset)
        return square.convert("RGB")  # Convert back to RGB for compatibility
    
    elif method == "fit_blur":
        # Fit image with blurred background of the same image
        # Create blurred background
        bg = image.copy()
        bg = bg.resize((thumb_px, thumb_px), Image.Resampling.LANCZOS)
        bg = bg.filter(Image.ImageFilter.GaussianBlur(radius=20))
        
        # Create thumbnail maintaining aspect ratio
        thumb = image.copy()
        thumb.thumbnail((thumb_px, thumb_px), Image.Resampling.LANCZOS)
        
        # Paste thumbnail on blurred background
        offset = ((thumb_px - thumb.width) // 2, (thumb_px - thumb.height) // 2)
        bg.paste(thumb, offset)
        return bg
    
    elif method == "fit_extend":
        # Extend edges of the image to fill background
        thumb = image.copy()
        thumb.thumbnail((thumb_px, thumb_px), Image.Resampling.LANCZOS)
        
        if thumb.width == thumb_px and thumb.height == thumb_px:
            return thumb
        
        # Create background by extending edges
        square = Image.new("RGB", (thumb_px, thumb_px))
        
        # Get dominant edge color
        try:
            edge_color = get_dominant_background_color(thumb)
            square = Image.new("RGB", (thumb_px, thumb_px), edge_color)
        except:
            # Fallback to stretching the image as background
            bg = image.resize((thumb_px, thumb_px), Image.Resampling.LANCZOS)
            square.paste(bg, (0, 0))
        
        # Paste the properly sized thumbnail on top
        offset = ((thumb_px - thumb.width) // 2, (thumb_px - thumb.height) // 2)
        square.paste(thumb, offset)
        return square
    
    else:
        # Default fallback - smart crop
        return smart_crop_center(image, thumb_px)

def build_sprite(thumb_px: int, out_path: str, input_folder: str = ".", method: str = "crop"):
    """
    Build a sprite sheet optimized for TensorFlow Projector
    
    Args:
        thumb_px: Size of each thumbnail (square)
        out_path: Output path for the sprite image
        input_folder: Folder containing the images to process
        method: Thumbnail creation method ("crop", "fit_transparent", "fit_blur", "fit_extend")
    """
    # Supported image extensions
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}
    
    # Get input directory
    input_dir = Path(input_folder)
    if not input_dir.exists():
        sys.exit(f"Input folder '{input_folder}' not found.")
    
    # Find all image files (non-recursive, same as your main script)
    files = []
    for file_path in input_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in exts:
            files.append(file_path)
    
    if not files:
        sys.exit(f"No images found in '{input_folder}'.")
    
    # Sort files by name for consistent ordering (same as your main script)
    files = sorted(files)
    
    print(f"Found {len(files)} images to process into sprite")
    print(f"Using method: {method}")
    
    method_descriptions = {
        "crop": "Smart center crop (no background, fills thumbnail)",
        "fit_transparent": "Fit with transparent background",
        "fit_blur": "Fit with blurred background from same image", 
        "fit_extend": "Fit with extended edge colors"
    }
    print(f"Method description: {method_descriptions.get(method, 'Unknown method')}")

    # Process images with progress bar
    thumbs = []
    skipped = 0
    
    print("Creating thumbnails...")
    for p in tqdm(files, desc="Processing images"):
        try:
            # Open and convert to RGB
            img = Image.open(p).convert("RGB")
            
            # Create thumbnail using selected method
            thumbnail = create_thumbnail_no_background(img, thumb_px, method)
            thumbs.append(thumbnail)
            
        except Exception as e:
            print(f"WARNING: Skipping {p.name}: {e}")
            skipped += 1
            continue

    if not thumbs:
        sys.exit("ERROR: No valid images could be processed.")
    
    n = len(thumbs)
    print(f"Successfully processed {n} images ({skipped} skipped)")
    
    # Calculate optimal grid dimensions
    # For TensorFlow Projector, we want a roughly square grid
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    
    print(f"Creating sprite grid: {cols} columns x {rows} rows")
    print(f"Sprite dimensions: {cols * thumb_px} x {rows * thumb_px} pixels")
    
    # Create the sprite sheet with transparent background if method supports it
    sprite_width = cols * thumb_px
    sprite_height = rows * thumb_px
    
    # Use black background for better contrast in latent space visualization
    bg_color = (0, 0, 0) if method == "crop" else (32, 32, 32)  # Dark gray for other methods
    sprite = Image.new("RGB", (sprite_width, sprite_height), bg_color)
    
    print("Building sprite sheet...")
    for i, tile in enumerate(tqdm(thumbs, desc="Placing thumbnails")):
        col = i % cols
        row = i // cols
        x = col * thumb_px
        y = row * thumb_px
        sprite.paste(tile, (x, y))
    
    # Save the sprite
    print(f"Saving sprite to: {out_path}")
    sprite.save(out_path, optimize=True, quality=95)
    
    # Print summary
    print("\n" + "="*50)
    print("Sprite creation complete!")
    print(f"Sprite file: {out_path}")
    print(f"Dimensions: {sprite_width} x {sprite_height} pixels")
    print(f"Thumbnails: {n} images ({thumb_px}x{thumb_px} each)")
    print(f"Grid layout: {cols} columns x {rows} rows")
    print(f"Method used: {method}")
    
    # File size info
    if os.path.exists(out_path):
        file_size = os.path.getsize(out_path) / (1024 * 1024)  # MB
        print(f"File size: {file_size:.1f} MB")
    
    print("\nReady for TensorFlow Projector!")
    print("Next steps:")
    print("   1. Upload vectors.tsv, metadata.tsv, and sprite.png to projector.tensorflow.org")
    print("   2. Configure the sprite image in the projector settings")
    print("   3. Explore your image embeddings!")
    
    if method != "crop":
        print(f"\nTip: For latent space visualization, 'crop' method usually works best")
        print(f"    as it eliminates background distractions and focuses on image content.")
    
    return {
        'total_images': n,
        'grid_cols': cols,
        'grid_rows': rows,
        'sprite_width': sprite_width,
        'sprite_height': sprite_height,
        'thumbnail_size': thumb_px,
        'method_used': method
    }

def validate_sprite_for_projector(sprite_info, expected_count):
    """
    Validate that the sprite is properly formatted for TensorFlow Projector
    """
    print(f"\nValidating sprite for TensorFlow Projector...")
    
    issues = []
    
    # Check if image count matches expected
    if sprite_info['total_images'] != expected_count:
        issues.append(f"Image count mismatch: sprite has {sprite_info['total_images']}, expected {expected_count}")
    
    # Check thumbnail size (TensorFlow Projector works best with certain sizes)
    recommended_sizes = [32, 64, 128, 256]
    if sprite_info['thumbnail_size'] not in recommended_sizes:
        issues.append(f"Thumbnail size {sprite_info['thumbnail_size']} not in recommended sizes: {recommended_sizes}")
    
    # Check sprite dimensions (shouldn't be too large for web display)
    max_dimension = 8192  # Reasonable limit for web browsers
    if sprite_info['sprite_width'] > max_dimension or sprite_info['sprite_height'] > max_dimension:
        issues.append(f"Sprite too large: {sprite_info['sprite_width']}×{sprite_info['sprite_height']} (max recommended: {max_dimension})")
    
    if issues:
        print("WARNING: Potential issues found:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("Sprite validation passed!")
    
    return len(issues) == 0

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Create optimized sprite sheets for TensorFlow Projector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sprite_transformer.py --input_folder "Photos_compressed" --size 64
  python sprite_transformer.py --input_folder "Photos_compressed" --size 128 --out "my_sprite.png"
  python sprite_transformer.py --input_folder "Photos_compressed" --size 64 --method crop
        """
    )
    ap.add_argument("--size", type=int, default=64, 
                   help="Size of each thumbnail in pixels (default: 64)")
    ap.add_argument("--out", default="sprite.png", 
                   help="Output filename for the sprite (default: sprite.png)")
    ap.add_argument("--input_folder", default=".", 
                   help="Folder containing images to process (default: current directory)")
    ap.add_argument("--expected_count", type=int, default=None,
                   help="Expected number of images for validation (optional)")
    ap.add_argument("--method", default="crop", 
                   choices=["crop", "fit_transparent", "fit_blur", "fit_extend"],
                   help="Thumbnail creation method: 'crop' (smart crop, no background), 'fit_transparent' (fit with transparent bg), 'fit_blur' (fit with blurred bg), 'fit_extend' (fit with extended edge colors)")
    
    args = ap.parse_args()
    
    # Build the sprite
    sprite_info = build_sprite(args.size, args.out, args.input_folder, args.method)
    
    # Validate if expected count provided
    if args.expected_count:
        validate_sprite_for_projector(sprite_info, args.expected_count)
