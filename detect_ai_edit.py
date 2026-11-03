import os
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import exifread
from transformers import CLIPProcessor, CLIPModel
import torch

# --- LAYER 1: ERROR LEVEL ANALYSIS (ELA) ---
# ELA detects if an image has been heavily compressed or edited by comparing it to a re-saved version.
def compute_ela(image_path, quality=90):
    try:
        original = Image.open(image_path).convert('RGB')
        temp_path = "temp_ela.jpg"
        original.save(temp_path, quality=quality)
        compressed = Image.open(temp_path)
        diff = ImageChops.difference(original, compressed)
        
        # Enhance the difference to see it visually
        extrema = diff.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0:
            max_diff = 1
        scale = 255.0 / max_diff
        diff = diff.point(lambda p: p * scale)
        
        os.remove(temp_path)
        
        # Calculate the ELA score (mean difference)
        ela_score = np.mean(np.array(diff))
        return ela_score
    except Exception as e:
        return 0.0

# --- LAYER 2: CLIP AI DETECTION (Zero-shot detection) ---
# We use CLIP to ask if the image looks like an AI generation.
def detect_ai_clip(image_path):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    image = Image.open(image_path).convert("RGB")
    
    # We present two hypotheses to the model
    inputs = processor(text=["A real photograph.", "An AI generated image."], images=image, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
    
    # Return probability of it being "AI generated"
    return probs[0][1].item() * 100

# --- LAYER 3: METADATA CHECK (Editing History) ---
def check_metadata(image_path):
    with open(image_path, 'rb') as f:
        tags = exifread.process_file(f)
    
    editing_flags = []
    
    # Check for Photoshop/Lightroom traces
    if 'Image Software' in tags:
        software = str(tags['Image Software'])
        if any(x in software.lower() for x in ['photoshop', 'lightroom', 'snapseed', 'picsart', 'facetune']):
            editing_flags.append(f"Edited in: {software}")
    
    # Check for AI tool histories (some AI tools leave metadata)
    if 'Image History' in tags:
        history = str(tags['Image History'])
        if 'ai' in history.lower() or 'generative' in history.lower():
            editing_flags.append(f"AI Tool detected in history: {history}")

    return editing_flags if editing_flags else ["No obvious editing software found."]

# --- MAIN SCANNER ---
def scan_image(image_path):
    print(f"\n--- Scanning Image: {os.path.basename(image_path)} ---")
    
    # 1. Check Metadata (Easiest)
    meta_flags = check_metadata(image_path)
    print(f"1. Metadata: {', '.join(meta_flags)}")
    
    # 2. Run ELA (For editing/compression)
    ela_score = compute_ela(image_path)
    print(f"2. Error Level Analysis (ELA) Score: {ela_score:.2f}")
    # Lower score (< 10) means it is highly compressed or heavily saved/edited. 
    # High score (> 50) means it's closer to original, raw quality (less edited).
    if ela_score < 20:
        print("   ➜ Result: WARNING - High compression/editing detected (The filter/smoothing is extreme).")
    elif 20 <= ela_score <= 50:
        print("   ➜ Result: Moderate edits detected.")
    else:
        print("   ➜ Result: Low compression, likely standard photo.")

    # 3. Run CLIP AI Check
    try:
        ai_prob = detect_ai_clip(image_path)
        print(f"3. AI Detection Probability: {ai_prob:.2f}%")
        if ai_prob > 60:
            print("   ➜ Result: HIGH probability of AI Generation.")
        elif 40 <= ai_prob <= 60:
            print("   ➜ Result: INCONCLUSIVE. Looks like a mix (Heavy filter/editing).")
        else:
            print("   ➜ Result: Low AI probability. Looks like a real photo.")
    except Exception as e:
        print(f"3. AI Detection: Could not run (likely due to torch/GPU). Error: {e}")

    print("-" * 40)

# --- RUN THE SCANNER ---
# Replace 'your_image.jpg' with your actual file path
scan_image('filterman.jpeg')