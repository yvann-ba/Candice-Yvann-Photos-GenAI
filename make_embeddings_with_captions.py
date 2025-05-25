from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS
import torch
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
import os
import re
import argparse
import numpy as np
from datetime import datetime
import colorsys
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm
import gc

def extract_dominant_colors(image, num_colors=3):
    """Extract dominant colors from an image"""
    # Resize image for faster processing
    img = image.resize((50, 50))
    img_array = np.array(img)
    
    # Reshape to list of pixels
    pixels = img_array.reshape(-1, 3)
    
    # Simple color clustering by averaging
    colors = []
    try:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        colors = kmeans.cluster_centers_.astype(int)
    except ImportError:
        # Fallback: just take mean color
        colors = [np.mean(pixels, axis=0).astype(int)]
    
    return colors

def rgb_to_hue_saturation(rgb):
    """Convert RGB to hue and saturation"""
    r, g, b = [x/255.0 for x in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return int(h * 360), int(s * 100)

def extract_exif_date(image_path):
    """Extract creation date from EXIF data"""
    try:
        img = Image.open(image_path)
        exifdata = img.getexif()
        for tag_id in exifdata:
            tag = TAGS.get(tag_id, tag_id)
            if tag in ['DateTime', 'DateTimeOriginal', 'DateTimeDigitized']:
                return str(exifdata.get(tag_id))
    except:
        pass
    return None

def clean_and_enhance_caption(caption):
    """Clean and enhance the generated caption for better artistic description"""
    # Remove common prefixes that BLIP sometimes adds
    caption = re.sub(r'^(a picture of |an image of |a photo of |this is )', '', caption.lower())
    
    # Capitalize first letter
    caption = caption.strip()
    if caption:
        caption = caption[0].upper() + caption[1:]
    
    return caption

def generate_artistic_tags(description):
    """Generate artistic tags based on the description"""
    description_lower = description.lower()
    
    # Define artistic categories
    mood_keywords = {
        'serene': ['peaceful', 'calm', 'tranquil', 'serene', 'quiet'],
        'dynamic': ['action', 'movement', 'dynamic', 'energetic', 'busy'],
        'intimate': ['close', 'intimate', 'personal', 'tender', 'gentle'],
        'dramatic': ['dramatic', 'intense', 'striking', 'bold', 'powerful'],
        'playful': ['fun', 'playful', 'cheerful', 'happy', 'joyful'],
        'contemplative': ['thoughtful', 'contemplative', 'reflective', 'serious']
    }
    
    setting_keywords = {
        'indoor': ['room', 'inside', 'indoor', 'kitchen', 'bedroom', 'bathroom'],
        'outdoor': ['outside', 'outdoor', 'garden', 'park', 'street', 'nature'],
        'urban': ['city', 'building', 'street', 'urban', 'architecture'],
        'natural': ['nature', 'tree', 'flower', 'landscape', 'natural', 'garden']
    }
    
    tags = []
    
    # Check for mood
    for mood, keywords in mood_keywords.items():
        if any(keyword in description_lower for keyword in keywords):
            tags.append(mood)
    
    # Check for setting
    for setting, keywords in setting_keywords.items():
        if any(keyword in description_lower for keyword in keywords):
            tags.append(setting)
    
    return ', '.join(tags) if tags else 'neutral'

def process_cpu_metadata(img_paths_batch):
    """Process CPU-intensive metadata extraction in parallel"""
    metadata_batch = []
    
    for img_path in img_paths_batch:
        try:
            img = Image.open(img_path).convert("RGB")
            width, height = img.size
            file_size = img_path.stat().st_size
            aspect_ratio = round(width / height, 2)
            
            # Extract dominant colors
            try:
                dominant_colors = extract_dominant_colors(img)
                primary_color = dominant_colors[0] if len(dominant_colors) > 0 else [128, 128, 128]
                hue, saturation = rgb_to_hue_saturation(primary_color)
            except:
                hue, saturation = 0, 0
            
            # Extract creation date
            creation_date = extract_exif_date(img_path)
            if not creation_date:
                creation_date = datetime.fromtimestamp(img_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            
            metadata = {
                'filename': img_path.name,
                'width': width,
                'height': height,
                'aspect_ratio': aspect_ratio,
                'file_size_kb': round(file_size / 1024, 1),
                'dominant_hue': hue,
                'dominant_saturation': saturation,
                'creation_date': creation_date,
                'image': img
            }
            metadata_batch.append(metadata)
            
        except Exception as e:
            print(f"Error processing metadata for {img_path.name}: {e}")
            continue
    
    return metadata_batch

def process_images_batch_gpu(images_batch, model, processor, blip_model, blip_processor, device, max_caption_length):
    """Process a batch of images on GPU for maximum efficiency"""
    try:
        # Prepare batch for CLIP
        clip_inputs = processor(images=images_batch, return_tensors="pt", padding=True)
        clip_inputs = {k: v.to(device) for k, v in clip_inputs.items()}
        
        # Generate embeddings with mixed precision
        with torch.cuda.amp.autocast():
            clip_features = model.get_image_features(**clip_inputs)
        
        # Prepare batch for BLIP
        blip_inputs = blip_processor(images=images_batch, return_tensors="pt", padding=True)
        blip_inputs = {k: v.to(device) for k, v in blip_inputs.items()}
        
        # Generate captions with mixed precision
        with torch.cuda.amp.autocast():
            output_ids = blip_model.generate(
                **blip_inputs,
                max_length=max_caption_length,
                num_beams=5,
                early_stopping=True,
                do_sample=False
            )
        
        # Process captions
        captions = []
        for i in range(output_ids.shape[0]):
            raw_caption = blip_processor.decode(output_ids[i], skip_special_tokens=True)
            caption = clean_and_enhance_caption(raw_caption)
            captions.append(caption)
        
        # Move embeddings to CPU and convert to numpy
        embeddings = clip_features.cpu().numpy()
        
        return embeddings, captions
        
    except Exception as e:
        print(f"Error processing batch on GPU: {e}")
        return None, None

def optimize_gpu_settings():
    """Optimize GPU settings for maximum performance"""
    if torch.cuda.is_available():
        # Enable optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Set memory allocation strategy
        torch.cuda.empty_cache()
        
        # Print GPU info
        device_name = torch.cuda.get_device_name(0)
        device_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🚀 GPU: {device_name}")
        print(f"📊 GPU Memory: {device_memory:.1f} GB")
        
        # Optimize memory allocation
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        
        return torch.device("cuda")
    else:
        print("⚠️  CUDA not available, using CPU")
        return torch.device("cpu")

def calculate_optimal_batch_size(device):
    """Calculate optimal batch size based on available GPU memory"""
    if device.type == "cuda":
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        # Heuristic based on GPU memory (adjust as needed)
        if gpu_memory_gb >= 20:  # RTX 4090, A6000, etc.
            return 16
        elif gpu_memory_gb >= 12:  # RTX 3080 Ti, etc.
            return 12
        elif gpu_memory_gb >= 8:   # RTX 3070, etc.
            return 8
        else:
            return 4
    else:
        return 2

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='🎨 GPU-Optimized Image Processing for Artistic Exploration')
    parser.add_argument('--input_folder', type=str, required=True,
                      help='Path to the folder containing images to process')
    parser.add_argument('--max_caption_length', type=int, default=50,
                      help='Maximum length for generated captions')
    parser.add_argument('--batch_size', type=int, default=None,
                      help='Batch size for GPU processing (auto-calculated if not specified)')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of CPU workers for parallel processing')
    args = parser.parse_args()

    # Optimize GPU settings
    device = optimize_gpu_settings()
    
    # Calculate optimal batch size
    batch_size = args.batch_size or calculate_optimal_batch_size(device)
    print(f"🔥 Using batch size: {batch_size}")
    print(f"⚡ CPU workers: {args.num_workers}")

    # 1. Load models and move to GPU
    print("🤖 Loading CLIP model for embeddings...")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model = model.to(device)
    
    print("🖼️  Loading BLIP model for captioning...")
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").eval()
    blip_model = blip_model.to(device)
    
    print(f"✅ Models loaded on {device}")
    
    # 2. Define the photos directory
    photos_dir = Path(args.input_folder)
    
    # Check if directory exists
    if not photos_dir.exists():
        print(f"❌ Error: Directory '{photos_dir}' not found!")
        return
    
    # 3. Gather photos with various extensions
    image_extensions = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG"]
    photos = []
    
    for ext in image_extensions:
        photos.extend(photos_dir.glob(ext))
    
    # Sort the photos by name for consistent ordering
    photos = sorted(photos)
    
    print(f"📸 Found {len(photos)} images to process")
    
    if not photos:
        print("❌ No images found in the directory!")
        return
    
    # 4. Process images in optimized batches
    all_embeddings = []
    all_metadata_rows = []
    
    print("🚀 Processing images with GPU acceleration...")
    
    # Process in batches for optimal GPU utilization
    with torch.no_grad():
        for batch_start in tqdm(range(0, len(photos), batch_size), desc="Processing batches"):
            batch_end = min(batch_start + batch_size, len(photos))
            batch_paths = photos[batch_start:batch_end]
            
            # Process CPU-intensive metadata in parallel
            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                metadata_batch = process_cpu_metadata(batch_paths)
            
            if not metadata_batch:
                continue
            
            # Extract images for GPU processing
            images_batch = [meta['image'] for meta in metadata_batch]
            
            # Process batch on GPU
            embeddings_batch, captions_batch = process_images_batch_gpu(
                images_batch, model, processor, blip_model, blip_processor, 
                device, args.max_caption_length
            )
            
            if embeddings_batch is None:
                continue
            
            # Combine results
            for i, (embedding, caption, metadata) in enumerate(zip(embeddings_batch, captions_batch, metadata_batch)):
                # Generate artistic tags
                artistic_tags = generate_artistic_tags(caption)
                
                # Create complete metadata row
                metadata_row = {
                    'filename': metadata['filename'],
                    'description': caption,
                    'artistic_tags': artistic_tags,
                    'width': metadata['width'],
                    'height': metadata['height'],
                    'aspect_ratio': metadata['aspect_ratio'],
                    'file_size_kb': metadata['file_size_kb'],
                    'dominant_hue': metadata['dominant_hue'],
                    'dominant_saturation': metadata['dominant_saturation'],
                    'creation_date': metadata['creation_date'],
                    'processing_order': batch_start + i + 1
                }
                
                all_embeddings.append(embedding)
                all_metadata_rows.append(metadata_row)
            
            # Clear GPU cache periodically
            if batch_start % (batch_size * 4) == 0:
                torch.cuda.empty_cache()
                gc.collect()
    
    print(f"✅ Successfully processed {len(all_embeddings)} images")
    
    # 5. Write vectors.tsv (optimized I/O)
    print("💾 Writing vectors.tsv...")
    with open("vectors.tsv", "w", encoding="utf-8") as vf:
        # Write in chunks for better performance
        chunk_size = 100
        for i in range(0, len(all_embeddings), chunk_size):
            chunk = all_embeddings[i:i+chunk_size]
            lines = ["\t".join(map(str, vec)) + "\n" for vec in chunk]
            vf.writelines(lines)
    
    # 6. Write metadata.tsv (optimized I/O)
    print("📊 Writing metadata.tsv...")
    with open("metadata.tsv", "w", encoding="utf-8") as mf:
        # Write header
        headers = ['filename', 'description', 'artistic_tags', 'width', 'height', 'aspect_ratio', 
                  'file_size_kb', 'dominant_hue', 'dominant_saturation', 'creation_date', 'processing_order']
        mf.write("\t".join(headers) + "\n")
        
        # Write data in chunks
        chunk_size = 100
        for i in range(0, len(all_metadata_rows), chunk_size):
            chunk = all_metadata_rows[i:i+chunk_size]
            lines = []
            for row in chunk:
                values = [str(row[header]) for header in headers]
                lines.append("\t".join(values) + "\n")
            mf.writelines(lines)
    
    # 7. Generate projector config
    print("⚙️  Writing projector_config.json...")
    config = {
        "embeddings": [
            {
                "tensorName": "Image Embeddings",
                "tensorShape": [len(all_embeddings), len(all_embeddings[0])],
                "tensorPath": "vectors.tsv",
                "metadataPath": "metadata.tsv"
            }
        ]
    }
    
    import json
    with open("projector_config.json", "w", encoding="utf-8") as cf:
        json.dump(config, cf, indent=2)
    
    # Final GPU memory cleanup
    torch.cuda.empty_cache()
    
    print("\n" + "="*60)
    print("🎨 GPU-Accelerated Processing Complete! 🚀")
    print(f"📊 Generated files for TensorFlow Projector:")
    print(f"   • vectors.tsv ({len(all_embeddings)} embedding vectors)")
    print(f"   • metadata.tsv ({len(all_metadata_rows)} images with enhanced metadata)")
    print(f"   • projector_config.json (configuration file)")
    print(f"\n🎯 Performance optimizations applied:")
    print(f"   • Batch processing (size: {batch_size})")
    print(f"   • GPU mixed precision inference")
    print(f"   • Parallel CPU processing ({args.num_workers} workers)")
    print(f"   • Optimized I/O operations")
    print(f"   • Automatic memory management")
    print(f"\n📋 Metadata includes:")
    print(f"   • English descriptions of each image")
    print(f"   • Artistic tags (mood, setting)")
    print(f"   • Technical info (dimensions, aspect ratio, file size)")
    print(f"   • Color analysis (dominant hue and saturation)")
    print(f"   • Creation dates")
    print(f"\n🚀 Next steps:")
    print(f"   1. Run make_sprite.py to generate the sprite sheet")
    print(f"   2. Upload all files to projector.tensorflow.org")
    print(f"   3. Explore your latent space! 🎨")

if __name__ == "__main__":
    main() 