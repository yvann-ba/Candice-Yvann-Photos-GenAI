#!/usr/bin/env python3
"""
Simple sprite creator with center crop
Creates a sprite sheet by center cropping all images to the same size
"""

from pathlib import Path
from PIL import Image, ImageOps
import math, argparse, sys
from tqdm import tqdm
import os

def center_crop_image(image, size):
    """
    Center crop image to exact square size
    This removes white borders and ensures consistent sizing
    """
    return ImageOps.fit(image, (size, size), Image.Resampling.LANCZOS)

def create_sprite_simple(input_folder, output_file="sprite.png", thumbnail_size=128):
    """
    Create sprite sheet with center-cropped images
    
    Args:
        input_folder: Folder containing images
        output_file: Output sprite filename
        thumbnail_size: Size of each thumbnail (square)
    """
    # Supported image extensions
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}
    
    # Get input directory
    input_dir = Path(input_folder)
    if not input_dir.exists():
        sys.exit(f"ERROR: Input folder '{input_folder}' not found.")
    
    # Find all image files
    files = []
    for file_path in input_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in exts:
            files.append(file_path)
    
    if not files:
        sys.exit(f"ERROR: No images found in '{input_folder}'.")
    
    # Sort files for consistent ordering
    files = sorted(files)
    
    print(f"Found {len(files)} images to process")
    print(f"Thumbnail size: {thumbnail_size}x{thumbnail_size} pixels")
    print(f"Method: Center crop (no white borders)")
    
    # Process images
    thumbnails = []
    skipped = 0
    
    print("Processing images...")
    for file_path in tqdm(files, desc="Creating thumbnails"):
        try:
            # Open and convert to RGB
            img = Image.open(file_path).convert("RGB")
            
            # Center crop to exact size
            thumbnail = center_crop_image(img, thumbnail_size)
            thumbnails.append(thumbnail)
            
        except Exception as e:
            print(f"WARNING: Skipping {file_path.name}: {e}")
            skipped += 1
            continue

    if not thumbnails:
        sys.exit("ERROR: No valid images could be processed.")
    
    n = len(thumbnails)
    print(f"Successfully processed {n} images ({skipped} skipped)")
    
    # Calculate grid dimensions
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    
    print(f"Grid layout: {cols} columns x {rows} rows")
    
    # Create sprite sheet
    sprite_width = cols * thumbnail_size
    sprite_height = rows * thumbnail_size
    
    print(f"Sprite dimensions: {sprite_width} x {sprite_height} pixels")
    
    # Create black background sprite
    sprite = Image.new("RGB", (sprite_width, sprite_height), (0, 0, 0))
    
    print("Building sprite sheet...")
    for i, thumbnail in enumerate(tqdm(thumbnails, desc="Placing thumbnails")):
        col = i % cols
        row = i // cols
        x = col * thumbnail_size
        y = row * thumbnail_size
        sprite.paste(thumbnail, (x, y))
    
    # Save sprite
    print(f"Saving sprite to: {output_file}")
    sprite.save(output_file, optimize=True, quality=95)
    
    # File size info
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
        print(f"File size: {file_size:.1f} MB")
    
    print("\n" + "="*50)
    print("SPRITE CREATION COMPLETE!")
    print(f"Output file: {output_file}")
    print(f"Total images: {n}")
    print(f"Thumbnail size: {thumbnail_size}x{thumbnail_size} pixels")
    print(f"Grid: {cols}x{rows}")
    print(f"Final sprite: {sprite_width}x{sprite_height} pixels")
    print("\nReady for TensorFlow Projector!")
    
    return {
        'total_images': n,
        'grid_cols': cols,
        'grid_rows': rows,
        'sprite_width': sprite_width,
        'sprite_height': sprite_height,
        'thumbnail_size': thumbnail_size,
        'output_file': output_file
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple sprite creator with center crop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python simple_sprite_crop.py --input "Photos_compressed" --size 128
  python simple_sprite_crop.py --input "Photos_compressed" --size 64 --output "my_sprite.png"
        """
    )
    
    parser.add_argument("--input", default="Photos_compressed",
                       help="Input folder containing images (default: Photos_compressed)")
    parser.add_argument("--size", type=int, default=128,
                       help="Thumbnail size in pixels (default: 128)")
    parser.add_argument("--output", default="sprite.png",
                       help="Output sprite filename (default: sprite.png)")
    
    args = parser.parse_args()
    
    # Validate input folder
    if not Path(args.input).exists():
        print(f"ERROR: Input folder '{args.input}' not found!")
        sys.exit(1)
    
    # Create sprite
    result = create_sprite_simple(args.input, args.output, args.size)
    print(f"\nSUCCESS! Created {result['output_file']} with {result['total_images']} images") 