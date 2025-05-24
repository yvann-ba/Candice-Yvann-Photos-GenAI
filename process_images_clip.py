from pathlib import Path
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import os

def main():
    # 1. Load CLIP model
    print("Loading CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    # 2. Define the photos directory
    photos_dir = Path("wetransfer_banque-d-images-2024-2025_2025-05-22_0809")
    
    # Check if directory exists
    if not photos_dir.exists():
        print(f"Error: Directory '{photos_dir}' not found!")
        return
    
    # 3. Gather photos with various extensions
    image_extensions = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG"]
    photos = []
    
    for ext in image_extensions:
        photos.extend(photos_dir.glob(ext))
    
    # Sort the photos by name
    photos = sorted(photos)
    
    print(f"Found {len(photos)} images to process")
    
    if not photos:
        print("No images found in the directory!")
        return
    
    # 4. Compute embeddings
    embeddings = []
    filenames = []
    
    print("Processing images...")
    with torch.no_grad():
        for i, img_path in enumerate(photos):
            try:
                print(f"Processing {i+1}/{len(photos)}: {img_path.name}")
                
                # Open and convert image to RGB
                img = Image.open(img_path).convert("RGB")
                
                # Process with CLIP
                inputs = processor(images=img, return_tensors="pt")
                feat = model.get_image_features(**inputs).squeeze().cpu().numpy()
                
                embeddings.append(feat)
                filenames.append(img_path.name)
                
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")
                continue
    
    print(f"Successfully processed {len(embeddings)} images")
    
    # 5. Write vectors.tsv
    print("Writing vectors.tsv...")
    with open("vectors.tsv", "w", encoding="utf-8") as vf:
        for vec in embeddings:
            vf.write("\t".join(map(str, vec)) + "\n")
    
    # 6. Write metadata.tsv
    print("Writing metadata.tsv...")
    with open("metadata.tsv", "w", encoding="utf-8") as mf:
        mf.write("filename\n")
        for name in filenames:
            mf.write(f"{name}\n")
    
    print("Processing complete!")
    print(f"Generated files:")
    print(f"- vectors.tsv ({len(embeddings)} vectors)")
    print(f"- metadata.tsv ({len(filenames)} filenames)")

if __name__ == "__main__":
    main() 