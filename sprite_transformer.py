from pathlib import Path
from PIL import Image
import math, argparse, sys
from tqdm import tqdm
import os

def build_sprite(thumb_px: int, out_path: str, input_folder: str = "."):
    """
    Build a sprite sheet optimized for TensorFlow Projector
    
    Args:
        thumb_px: Size of each thumbnail (square)
        out_path: Output path for the sprite image
        input_folder: Folder containing the images to process
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
    
    print(f"📸 Found {len(files)} images to process into sprite")
    
    # Pillow version compatibility
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:          # Pillow ≤9
        resample = Image.LANCZOS

    # Process images with progress bar
    thumbs = []
    skipped = 0
    
    print("🖼️  Creating thumbnails...")
    for p in tqdm(files, desc="Processing images"):
        try:
            # Open and convert to RGB
            img = Image.open(p).convert("RGB")
            
            # Create thumbnail maintaining aspect ratio
            img.thumbnail((thumb_px, thumb_px), resample=resample)

            # Create square canvas with white background
            square = Image.new("RGB", (thumb_px, thumb_px), (255, 255, 255))
            
            # Center the thumbnail in the square
            offset = ((thumb_px - img.width) // 2, (thumb_px - img.height) // 2)
            square.paste(img, offset)
            
            thumbs.append(square)
            
        except Exception as e:
            print(f"⚠️  Skipping {p.name}: {e}")
            skipped += 1
            continue

    if not thumbs:
        sys.exit("❌ No valid images could be processed.")
    
    n = len(thumbs)
    print(f"✅ Successfully processed {n} images ({skipped} skipped)")
    
    # Calculate optimal grid dimensions
    # For TensorFlow Projector, we want a roughly square grid
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    
    print(f"📐 Creating sprite grid: {cols} columns × {rows} rows")
    print(f"🎯 Sprite dimensions: {cols * thumb_px} × {rows * thumb_px} pixels")
    
    # Create the sprite sheet
    sprite_width = cols * thumb_px
    sprite_height = rows * thumb_px
    sprite = Image.new("RGB", (sprite_width, sprite_height), (255, 255, 255))
    
    print("🎨 Building sprite sheet...")
    for i, tile in enumerate(tqdm(thumbs, desc="Placing thumbnails")):
        col = i % cols
        row = i // cols
        x = col * thumb_px
        y = row * thumb_px
        sprite.paste(tile, (x, y))
    
    # Save the sprite
    print(f"💾 Saving sprite to: {out_path}")
    sprite.save(out_path, optimize=True, quality=95)
    
    # Print summary
    print("\n" + "="*50)
    print("🎉 Sprite creation complete!")
    print(f"📊 Sprite file: {out_path}")
    print(f"📏 Dimensions: {sprite_width} × {sprite_height} pixels")
    print(f"🖼️  Thumbnails: {n} images ({thumb_px}×{thumb_px} each)")
    print(f"📐 Grid layout: {cols} columns × {rows} rows")
    
    # File size info
    if os.path.exists(out_path):
        file_size = os.path.getsize(out_path) / (1024 * 1024)  # MB
        print(f"📦 File size: {file_size:.1f} MB")
    
    print("\n🚀 Ready for TensorFlow Projector!")
    print("📋 Next steps:")
    print("   1. Upload vectors.tsv, metadata.tsv, and sprite.png to projector.tensorflow.org")
    print("   2. Configure the sprite image in the projector settings")
    print("   3. Explore your image embeddings! 🎨")
    
    return {
        'total_images': n,
        'grid_cols': cols,
        'grid_rows': rows,
        'sprite_width': sprite_width,
        'sprite_height': sprite_height,
        'thumbnail_size': thumb_px
    }

def validate_sprite_for_projector(sprite_info, expected_count):
    """
    Validate that the sprite is properly formatted for TensorFlow Projector
    """
    print(f"\n🔍 Validating sprite for TensorFlow Projector...")
    
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
        print("⚠️  Potential issues found:")
        for issue in issues:
            print(f"   • {issue}")
    else:
        print("✅ Sprite validation passed!")
    
    return len(issues) == 0

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="🎨 Create optimized sprite sheets for TensorFlow Projector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sprite_transformer.py --input_folder "Photos_compressed" --size 64
  python sprite_transformer.py --input_folder "Photos_compressed" --size 128 --out "my_sprite.png"
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
    
    args = ap.parse_args()
    
    # Build the sprite
    sprite_info = build_sprite(args.size, args.out, args.input_folder)
    
    # Validate if expected count provided
    if args.expected_count:
        validate_sprite_for_projector(sprite_info, args.expected_count)
