# CLIP Image Processing Script

This script processes images from the `wetransfer_banque-d-images-2024-2025_2025-05-22_0809` folder using OpenAI's CLIP model to generate image embeddings.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure the image folder exists:**
   Make sure the `wetransfer_banque-d-images-2024-2025_2025-05-22_0809` folder is in the same directory as the script.

## Usage

Run the script:
```bash
python process_images_clip.py
```

## What the script does

1. **Loads the CLIP model:** Uses OpenAI's CLIP ViT-Large model for image feature extraction
2. **Finds images:** Searches for JPG and JPEG files in the specified folder
3. **Processes images:** Converts each image to RGB and extracts CLIP features
4. **Generates output files:**
   - `vectors.tsv`: Contains the image embeddings (one vector per line, tab-separated)
   - `metadata.tsv`: Contains the corresponding filenames

## Output

- **vectors.tsv**: Each line contains the CLIP embedding for one image (768-dimensional vector)
- **metadata.tsv**: Each line contains the filename corresponding to the vector in the same row

## Supported formats

- JPG/JPEG files
- The script automatically skips ARW files (raw format) as they require special handling

## Error handling

The script includes error handling to skip problematic images and continue processing the rest.

## Note

The first run will download the CLIP model (~1.7GB), so ensure you have a good internet connection and sufficient disk space. 