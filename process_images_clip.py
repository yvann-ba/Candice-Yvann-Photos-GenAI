from pathlib import Path
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
import os
import re
import argparse

def get_short_description(text, max_words=3):
    # Remove special characters and split into words
    words = re.findall(r'\b\w+\b', text.lower())
    # Take first max_words words
    return '_'.join(words[:max_words])

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process images with CLIP and BLIP models')
    parser.add_argument('--input_folder', type=str, required=True,
                      help='Path to the folder containing images to process')
    args = parser.parse_args()

    # 1. Load CLIP and BLIP models
    print("Loading CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    print("Loading BLIP model...")
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").eval()
    
    # 2. Define the photos directory
    photos_dir = Path(args.input_folder)
    
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
    
    # 4. Compute embeddings and generate descriptions
    embeddings = []
    filenames = []
    descriptions = []
    
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
                
                # Generate description with BLIP
                blip_inputs = blip_processor(img, return_tensors="pt")
                output_ids = blip_model.generate(**blip_inputs, max_length=30)
                description = blip_processor.decode(output_ids[0], skip_special_tokens=True)
                
                # Get short description for filename
                short_desc = get_short_description(description)
                
                # Create new filename with short description
                new_filename = f"{short_desc}{img_path.suffix}"
                new_path = img_path.parent / new_filename
                
                # Rename file
                img_path.rename(new_path)
                
                embeddings.append(feat)
                filenames.append(new_filename)
                descriptions.append(description)
                
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
        mf.write("filename\tdescription\n")
        for name, desc in zip(filenames, descriptions):
            mf.write(f"{name}\t{desc}\n")
    
    print("Processing complete!")
    print(f"Generated files:")
    print(f"- vectors.tsv ({len(embeddings)} vectors)")
    print(f"- metadata.tsv ({len(filenames)} filenames with descriptions)")

if __name__ == "__main__":
    main() 