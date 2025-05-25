from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS
import torch
from transformers import (
    AutoImageProcessor, AutoModel,  # For DINOv2
    InstructBlipProcessor, InstructBlipForConditionalGeneration  # For InstructBLIP
)
import os
import re
import argparse
import numpy as np
from datetime import datetime
import colorsys
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm
import gc
import random

def extract_dominant_colors(image, num_colors=5):
    """Extract dominant colors from an image with better analysis"""
    # Resize image for faster processing
    img = image.resize((100, 100))
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

def analyze_color_palette(colors):
    """Analyze color palette for artistic insights"""
    if len(colors) == 0:
        return "monochrome", "neutral", "balanced"
    
    # Convert to HSV for better analysis
    hsv_colors = []
    for color in colors:
        r, g, b = [x/255.0 for x in color]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        hsv_colors.append((h * 360, s * 100, v * 100))
    
    # Analyze color temperature
    warm_count = sum(1 for h, s, v in hsv_colors if (h < 60 or h > 300) and s > 20)
    cool_count = sum(1 for h, s, v in hsv_colors if 120 < h < 300 and s > 20)
    
    if warm_count > cool_count:
        temperature = "warm"
    elif cool_count > warm_count:
        temperature = "cool"
    else:
        temperature = "neutral"
    
    # Analyze saturation
    avg_saturation = np.mean([s for h, s, v in hsv_colors])
    if avg_saturation > 70:
        saturation_level = "vibrant"
    elif avg_saturation > 40:
        saturation_level = "moderate"
    else:
        saturation_level = "muted"
    
    # Analyze contrast
    values = [v for h, s, v in hsv_colors]
    contrast = max(values) - min(values)
    if contrast > 50:
        contrast_level = "high-contrast"
    elif contrast > 25:
        contrast_level = "balanced"
    else:
        contrast_level = "low-contrast"
    
    return temperature, saturation_level, contrast_level

def analyze_composition(image):
    """Analyze image composition"""
    width, height = image.size
    aspect_ratio = width / height
    
    # Basic composition analysis
    if abs(aspect_ratio - 1.0) < 0.1:
        format_type = "square"
    elif aspect_ratio > 1.5:
        format_type = "landscape"
    elif aspect_ratio < 0.7:
        format_type = "portrait"
    else:
        format_type = "standard"
    
    # Analyze image complexity (using edge detection simulation)
    img_array = np.array(image.convert('L'))
    
    # Simple edge detection approximation
    grad_x = np.abs(np.diff(img_array, axis=1))
    grad_y = np.abs(np.diff(img_array, axis=0))
    edge_density = (np.mean(grad_x) + np.mean(grad_y)) / 2
    
    if edge_density > 30:
        complexity = "complex"
    elif edge_density > 15:
        complexity = "moderate"
    else:
        complexity = "simple"
    
    return format_type, complexity

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
    # Remove repetitive and verbose prefixes
    caption = re.sub(r'^(the image depicts|the image features|the image shows|this image shows|this is an image of|in this image)', '', caption.lower())
    caption = re.sub(r'^(a picture of |an image of |a photo of |this is )', '', caption)
    
    # Remove repetitive sentences
    sentences = caption.split('.')
    unique_sentences = []
    seen = set()
    
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and sentence not in seen and len(sentence) > 10:
            unique_sentences.append(sentence)
            seen.add(sentence)
    
    # Join and clean up
    caption = '. '.join(unique_sentences[:2])  # Keep only first 2 unique sentences
    caption = re.sub(r'\s+', ' ', caption)  # Remove extra spaces
    caption = caption.strip()
    
    # Capitalize first letter
    if caption:
        caption = caption[0].upper() + caption[1:]
    
    return caption

def generate_sophisticated_artistic_tags(description, colors, composition_data, color_analysis):
    """Generate sophisticated and diverse artistic tags"""
    description_lower = description.lower()
    format_type, complexity = composition_data
    temperature, saturation_level, contrast_level = color_analysis
    
    tags = []
    
    # Color-based tags
    tags.append(f"color-{temperature}")
    tags.append(f"saturation-{saturation_level}")
    tags.append(f"contrast-{contrast_level}")
    
    # Composition tags
    tags.append(f"format-{format_type}")
    tags.append(f"complexity-{complexity}")
    
    # Advanced mood analysis
    mood_keywords = {
        'melancholic': ['shadow', 'dark', 'gloomy', 'sad', 'alone', 'empty', 'abandoned'],
        'euphoric': ['bright', 'joyful', 'celebration', 'party', 'laugh', 'smile', 'dancing'],
        'contemplative': ['thoughtful', 'quiet', 'peaceful', 'meditation', 'reflection', 'solitude'],
        'energetic': ['movement', 'action', 'dynamic', 'sports', 'running', 'jumping', 'busy'],
        'romantic': ['couple', 'kiss', 'love', 'hearts', 'wedding', 'flowers', 'sunset'],
        'mysterious': ['fog', 'mist', 'shadow', 'silhouette', 'hidden', 'secret', 'night'],
        'nostalgic': ['vintage', 'old', 'memory', 'past', 'sepia', 'faded', 'retro'],
        'dramatic': ['intense', 'powerful', 'storm', 'contrast', 'striking', 'bold'],
        'serene': ['calm', 'peaceful', 'tranquil', 'still', 'quiet', 'zen', 'minimal'],
        'playful': ['fun', 'colorful', 'child', 'toy', 'game', 'whimsical', 'cute']
    }
    
    # Subject matter analysis
    subject_keywords = {
        'portrait': ['person', 'face', 'people', 'human', 'man', 'woman', 'child'],
        'nature': ['tree', 'flower', 'plant', 'landscape', 'forest', 'mountain', 'river'],
        'urban': ['building', 'city', 'street', 'car', 'architecture', 'bridge', 'road'],
        'interior': ['room', 'indoor', 'furniture', 'house', 'home', 'kitchen', 'bedroom'],
        'food': ['eat', 'food', 'meal', 'restaurant', 'cooking', 'kitchen', 'plate'],
        'animals': ['dog', 'cat', 'bird', 'animal', 'pet', 'wildlife', 'horse'],
        'technology': ['computer', 'phone', 'screen', 'digital', 'electronic', 'device'],
        'art': ['painting', 'sculpture', 'gallery', 'artistic', 'creative', 'design'],
        'transportation': ['car', 'train', 'plane', 'boat', 'bicycle', 'vehicle', 'travel'],
        'events': ['party', 'wedding', 'celebration', 'festival', 'concert', 'gathering']
    }
    
    # Lighting analysis
    lighting_keywords = {
        'golden-hour': ['warm', 'sunset', 'sunrise', 'golden', 'soft light'],
        'dramatic-lighting': ['shadow', 'contrast', 'spotlight', 'dramatic'],
        'natural-light': ['daylight', 'outdoor', 'bright', 'sunny'],
        'artificial-light': ['indoor', 'lamp', 'neon', 'fluorescent'],
        'low-light': ['dark', 'night', 'dim', 'moody'],
        'backlit': ['silhouette', 'rim light', 'backlighting'],
        'soft-light': ['gentle', 'diffused', 'even lighting']
    }
    
    # Photography style analysis
    style_keywords = {
        'candid': ['natural', 'spontaneous', 'unposed', 'authentic'],
        'artistic': ['creative', 'abstract', 'stylized', 'artistic'],
        'documentary': ['real', 'authentic', 'story', 'moment'],
        'minimalist': ['simple', 'clean', 'minimal', 'sparse'],
        'vintage': ['old', 'retro', 'classic', 'aged'],
        'modern': ['contemporary', 'sleek', 'current', 'fresh'],
        'intimate': ['close', 'personal', 'private', 'tender']
    }
    
    # Apply sophisticated analysis
    for category, keywords in mood_keywords.items():
        if any(keyword in description_lower for keyword in keywords):
            tags.append(f"mood-{category}")
    
    for category, keywords in subject_keywords.items():
        if any(keyword in description_lower for keyword in keywords):
            tags.append(f"subject-{category}")
    
    for category, keywords in lighting_keywords.items():
        if any(keyword in description_lower for keyword in keywords):
            tags.append(f"lighting-{category}")
    
    for category, keywords in style_keywords.items():
        if any(keyword in description_lower for keyword in keywords):
            tags.append(f"style-{category}")
    
    # Remove duplicates and limit to most relevant tags
    unique_tags = list(dict.fromkeys(tags))
    
    # If no specific tags found, add some based on color analysis
    if len([t for t in unique_tags if not t.startswith(('color-', 'saturation-', 'contrast-', 'format-', 'complexity-'))]) < 2:
        if temperature == 'warm':
            unique_tags.append('mood-cozy')
        elif temperature == 'cool':
            unique_tags.append('mood-fresh')
        
        if saturation_level == 'vibrant':
            unique_tags.append('style-bold')
        elif saturation_level == 'muted':
            unique_tags.append('style-subtle')
    
    return ', '.join(unique_tags[:8])  # Limit to 8 most relevant tags

def process_cpu_metadata(img_paths_batch):
    """Process CPU-intensive metadata extraction in parallel with enhanced analysis"""
    metadata_batch = []
    
    for img_path in img_paths_batch:
        try:
            img = Image.open(img_path).convert("RGB")
            width, height = img.size
            file_size = img_path.stat().st_size
            aspect_ratio = round(width / height, 2)
            
            # Enhanced color analysis
            dominant_colors = extract_dominant_colors(img)
            primary_color = dominant_colors[0] if len(dominant_colors) > 0 else [128, 128, 128]
            hue, saturation = rgb_to_hue_saturation(primary_color)
            
            # Advanced color analysis
            color_temperature, saturation_level, contrast_level = analyze_color_palette(dominant_colors)
            
            # Composition analysis
            format_type, complexity = analyze_composition(img)
            
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
                'color_temperature': color_temperature,
                'saturation_level': saturation_level,
                'contrast_level': contrast_level,
                'format_type': format_type,
                'complexity': complexity,
                'image': img,
                'composition_data': (format_type, complexity),
                'color_analysis': (color_temperature, saturation_level, contrast_level)
            }
            metadata_batch.append(metadata)
            
        except Exception as e:
            print(f"Error processing metadata for {img_path.name}: {e}")
            continue
    
    return metadata_batch

def process_images_batch_gpu(images_batch, dinov2_model, dinov2_processor, instructblip_model, instructblip_processor, device, max_caption_length):
    """Process a batch of images on GPU with better prompts for more concise descriptions"""
    try:
        # Prepare batch for DINOv2 embeddings
        dinov2_inputs = dinov2_processor(images=images_batch, return_tensors="pt", padding=True)
        dinov2_inputs = {k: v.to(device) for k, v in dinov2_inputs.items()}
        
        # Generate embeddings with mixed precision using DINOv2
        with torch.cuda.amp.autocast():
            dinov2_outputs = dinov2_model(**dinov2_inputs)
            # Use the CLS token (first token) for global image representation
            embeddings = dinov2_outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        # Prepare batch for InstructBLIP captions with better prompts
        # Use varied prompts for more diverse and concise descriptions
        prompt_variations = [
            "Describe the key elements and subjects in this image concisely.",
            "What are the main visual elements and composition of this image?",
            "Identify the subjects, setting, and visual style of this image briefly.",
            "Summarize the content and mood of this image in one sentence.",
            "Describe the main subjects and atmosphere of this image."
        ]
        
        prompts = [random.choice(prompt_variations) for _ in range(len(images_batch))]
        instructblip_inputs = instructblip_processor(images=images_batch, text=prompts, return_tensors="pt", padding=True)
        instructblip_inputs = {k: v.to(device) for k, v in instructblip_inputs.items()}
        
        # Generate captions with mixed precision using InstructBLIP
        with torch.cuda.amp.autocast():
            output_ids = instructblip_model.generate(
                **instructblip_inputs,
                max_length=max_caption_length,
                num_beams=3,  # Reduced for more creativity
                early_stopping=True,
                do_sample=True,  # Enable sampling for variety
                temperature=0.7,  # Add some creativity
                repetition_penalty=2.0,  # Stronger penalty for repetition
                length_penalty=0.8,  # Encourage shorter responses
                no_repeat_ngram_size=3  # Prevent 3-gram repetition
            )
        
        # Process captions
        captions = []
        for i in range(output_ids.shape[0]):
            raw_caption = instructblip_processor.decode(output_ids[i], skip_special_tokens=True)
            caption = clean_and_enhance_caption(raw_caption)
            captions.append(caption)
        
        # Move embeddings to CPU and convert to numpy
        embeddings_np = embeddings.cpu().numpy()
        
        return embeddings_np, captions
        
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
        
        # Optimize memory allocation for high-end setup
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
        
        return torch.device("cuda")
    else:
        print("⚠️  CUDA not available, using CPU")
        return torch.device("cpu")

def calculate_optimal_batch_size(device):
    """Calculate optimal batch size for RTX 4090 with 24GB VRAM"""
    if device.type == "cuda":
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        # Optimized for RTX 4090 with 24GB VRAM
        if gpu_memory_gb >= 20:  # RTX 4090, A6000, etc.
            return 12  # Increased batch size for better performance
        elif gpu_memory_gb >= 16:  # RTX 4080, etc.
            return 8
        elif gpu_memory_gb >= 12:  # RTX 3080 Ti, etc.
            return 6
        else:
            return 4
    else:
        return 2

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='🎨 Advanced GPU-Optimized Image Processing with Rich Artistic Metadata')
    parser.add_argument('--input_folder', type=str, required=True,
                      help='Path to the folder containing images to process')
    parser.add_argument('--max_caption_length', type=int, default=150,
                      help='Maximum length for generated captions (reduced for conciseness)')
    parser.add_argument('--batch_size', type=int, default=None,
                      help='Batch size for GPU processing (auto-calculated if not specified)')
    parser.add_argument('--num_workers', type=int, default=8,
                      help='Number of CPU workers for parallel processing (increased for 64GB RAM)')
    args = parser.parse_args()

    # Optimize GPU settings
    device = optimize_gpu_settings()
    
    # Calculate optimal batch size
    batch_size = args.batch_size or calculate_optimal_batch_size(device)
    print(f"🔥 Using batch size: {batch_size} (optimized for RTX 4090)")
    print(f"⚡ CPU workers: {args.num_workers} (optimized for 64GB RAM)")

    # 1. Load DINOv2-Large for embeddings
    print("🤖 Loading DINOv2-Large model for embeddings...")
    dinov2_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
    dinov2_model = AutoModel.from_pretrained("facebook/dinov2-large").eval()
    dinov2_model = dinov2_model.to(device)
    
    # 2. Load InstructBLIP-FLan-T5-XL for captioning
    print("🖼️  Loading InstructBLIP-FLan-T5-XL model for enhanced captioning...")
    instructblip_processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")
    instructblip_model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl").eval()
    instructblip_model = instructblip_model.to(device)
    
    print(f"✅ Models loaded on {device}")
    
    # 3. Define the photos directory
    photos_dir = Path(args.input_folder)
    
    # Check if directory exists
    if not photos_dir.exists():
        print(f"❌ Error: Directory '{photos_dir}' not found!")
        return
    
    # 4. Gather photos with various extensions
    image_extensions = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
    photos = []
    
    for file_path in photos_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            photos.append(file_path)
    
    photos = sorted(photos)
    
    print(f"📸 Found {len(photos)} images to process")
    
    if not photos:
        print("❌ No images found in the directory!")
        return
    
    # 5. Create output directory
    output_dir = Path("embeddings_2")
    output_dir.mkdir(exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # 6. Process images in optimized batches
    all_embeddings = []
    all_metadata_rows = []
    
    print("🚀 Processing images with advanced artistic analysis...")
    
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
                images_batch, dinov2_model, dinov2_processor, 
                instructblip_model, instructblip_processor,
                device, args.max_caption_length
            )
            
            if embeddings_batch is None:
                continue
            
            # Combine results with enhanced metadata
            for i, (embedding, caption, metadata) in enumerate(zip(embeddings_batch, captions_batch, metadata_batch)):
                # Generate sophisticated artistic tags
                artistic_tags = generate_sophisticated_artistic_tags(
                    caption, 
                    metadata.get('dominant_colors', []), 
                    metadata['composition_data'], 
                    metadata['color_analysis']
                )
                
                # Create comprehensive metadata row
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
                    'color_temperature': metadata['color_temperature'],
                    'saturation_level': metadata['saturation_level'],
                    'contrast_level': metadata['contrast_level'],
                    'format_type': metadata['format_type'],
                    'complexity': metadata['complexity'],
                    'creation_date': metadata['creation_date'],
                    'processing_order': batch_start + i + 1
                }
                
                all_embeddings.append(embedding)
                all_metadata_rows.append(metadata_row)
            
            # Clear GPU cache periodically
            if batch_start % (batch_size * 3) == 0:
                torch.cuda.empty_cache()
                gc.collect()
    
    print(f"✅ Successfully processed {len(all_embeddings)} images")
    
    # 7. Write enhanced vectors.tsv
    vectors_file = output_dir / "vectors.tsv"
    print(f"💾 Writing {vectors_file}...")
    with open(vectors_file, "w", encoding="utf-8") as vf:
        chunk_size = 100
        for i in range(0, len(all_embeddings), chunk_size):
            chunk = all_embeddings[i:i+chunk_size]
            lines = ["\t".join(map(str, vec)) + "\n" for vec in chunk]
            vf.writelines(lines)
    
    # 8. Write enhanced metadata.tsv with rich artistic metadata
    metadata_file = output_dir / "metadata.tsv"
    print(f"📊 Writing enhanced {metadata_file}...")
    with open(metadata_file, "w", encoding="utf-8") as mf:
        # Enhanced headers with rich metadata
        headers = [
            'filename', 'description', 'artistic_tags', 'width', 'height', 'aspect_ratio', 
            'file_size_kb', 'dominant_hue', 'dominant_saturation', 'color_temperature',
            'saturation_level', 'contrast_level', 'format_type', 'complexity',
            'creation_date', 'processing_order'
        ]
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
    
    # 9. Generate enhanced projector config
    config_file = output_dir / "projector_config.json"
    print(f"⚙️  Writing enhanced {config_file}...")
    config = {
        "embeddings": [
            {
                "tensorName": "DINOv2 + Rich Artistic Metadata",
                "tensorShape": [len(all_embeddings), len(all_embeddings[0])],
                "tensorPath": "vectors.tsv",
                "metadataPath": "metadata.tsv"
            }
        ]
    }
    
    import json
    with open(config_file, "w", encoding="utf-8") as cf:
        json.dump(config, cf, indent=2)
    
    # Final GPU memory cleanup
    torch.cuda.empty_cache()
    
    print("\n" + "="*75)
    print("🎨 Advanced Artistic Analysis Complete! 🚀")
    print(f"📊 Generated files in {output_dir}/:")
    print(f"   • vectors.tsv ({len(all_embeddings)} DINOv2-Large embeddings)")
    print(f"   • metadata.tsv ({len(all_metadata_rows)} images with rich artistic metadata)")
    print(f"   • projector_config.json (enhanced configuration)")
    print(f"\n✨ Enhanced Features:")
    print(f"   • Concise, factual descriptions (no 'The image depicts')")
    print(f"   • Sophisticated artistic tags (mood, style, lighting, subject)")
    print(f"   • Advanced color analysis (temperature, saturation, contrast)")
    print(f"   • Composition analysis (format, complexity)")
    print(f"   • Diverse prompt variations for better descriptions")
    print(f"   • Optimized for RTX 4090 + 64GB RAM (batch: {batch_size}, workers: {args.num_workers})")
    print(f"\n🎯 Rich Metadata Fields:")
    print(f"   • Artistic mood and style classification")
    print(f"   • Color temperature and palette analysis")
    print(f"   • Composition and complexity metrics")
    print(f"   • Subject matter and lighting detection")
    print(f"   • Enhanced visual diversity analysis")
    print(f"\n🚀 Next steps:")
    print(f"   1. Run make_sprite.py to generate the sprite sheet")
    print(f"   2. Upload all files from {output_dir}/ to projector.tensorflow.org")
    print(f"   3. Explore your enhanced artistic latent space! 🎨")

if __name__ == "__main__":
    main() 