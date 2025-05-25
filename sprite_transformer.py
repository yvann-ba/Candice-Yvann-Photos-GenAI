from pathlib import Path
from PIL import Image
import math, argparse, sys

def build_sprite(thumb_px: int, out_path: str):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}
    files = [p for p in Path(".").iterdir() if p.suffix.lower() in exts]
    if not files:
        sys.exit("No images found.")

    # Pillow ≥10
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:          # Pillow ≤9
        resample = Image.LANCZOS

    thumbs = []
    for p in sorted(files):
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            print(f"Skipping {p.name}: {e}")
            continue
        img.thumbnail((thumb_px, thumb_px), resample=resample)

        square = Image.new("RGB", (thumb_px, thumb_px), (255, 255, 255))
        off = ((thumb_px - img.width)//2, (thumb_px - img.height)//2)
        square.paste(img, off)
        thumbs.append(square)

    n = len(thumbs)
    cols = math.isqrt(n) + 1
    rows = math.ceil(n / cols)

    sprite = Image.new("RGB", (cols*thumb_px, rows*thumb_px), (255, 255, 255))
    for i, tile in enumerate(thumbs):
        sprite.paste(tile, ((i % cols)*thumb_px, (i // cols)*thumb_px))

    sprite.save(out_path)
    print(f"Created {out_path} with {n} thumbs.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--out", default="sprite.png")
    args = ap.parse_args()
    build_sprite(args.size, args.out)
