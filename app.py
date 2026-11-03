import streamlit as st
import torch
import numpy as np
import cv2  # <--- REQUIRED FOR THE NEW TEXT DETECTOR
from PIL import Image, ImageChops
import exifread
import matplotlib.pyplot as plt
import io
import os
from transformers import CLIPProcessor, CLIPModel

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI & Editing Forensics", layout="wide")
st.title("🕵️ AI Generation & Editing Heatmap Detector")
st.write("Upload an image. The system checks for AI generation, heavy editing, and generates a heatmap showing potentially altered regions.")

# --- CACHED MODELS ---
@st.cache_resource
def load_clip_model():
    return CLIPModel.from_pretrained("openai/clip-vit-base-patch32"), CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# --- LAYER 1: ELA (Editing Detection) ---
def compute_ela(image_path):
    original = Image.open(image_path).convert('RGB')
    temp_path = "temp_ela_streamlit.jpg"
    original.save(temp_path, quality=85)
    compressed = Image.open(temp_path)
    diff = ImageChops.difference(original, compressed)
    ela_score = np.mean(np.array(diff))
    
    if os.path.exists(temp_path):
        os.remove(temp_path)
    return ela_score, diff

# --- LAYER 2: CLIP AI WHOLE-IMAGE DETECTION ---
def detect_ai_clip(image):
    model, processor = load_clip_model()
    inputs = processor(text=["A real photograph.", "An AI generated image."], images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)
    return probs[0][1].item() * 100

# --- LAYER 3: AI TEXT ARTIFACT DETECTOR ---
def detect_ai_text_artifacts(image_pil):
    # Convert PIL image to OpenCV format
    img = np.array(image_pil.convert('L'))
    
    # Apply a threshold to find bright, high-contrast "blobs" (which AI often uses for background text)
    _, binary = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)
    
    # Count the number of these bright, compressed pixels
    bright_pixels = np.sum(binary > 0)
    
    if bright_pixels > 1000: # Threshold for "there are suspicious blobs"
        return "⚠️ Suspicious: Blurry/Unreadable text or blobs detected in background (Common AI artifact)."
    return "✅ No obvious AI text artifacts detected."

# --- LAYER 4: AI SMOOTHNESS HEATMAP (UPDATED FOR AI) ---
def generate_ai_smoothness_heatmap(image_pil, grid_size=16):
    """
    AI images have unnaturally LOW standard deviation (super smooth).
    Real images have natural grain (HIGHER standard deviation).
    We invert the scale so SMOOTH areas GLOW YELLOW.
    """
    img = np.array(image_pil.convert('L'))
    h, w = img.shape
    heatmap = np.zeros((h, w))
    
    patch_h, patch_w = h // grid_size, w // grid_size
    
    for y in range(0, h - patch_h, patch_h):
        for x in range(0, w - patch_w, patch_w):
            patch = img[y:y+patch_h, x:x+patch_w]
            noise_level = np.std(patch)
            
            # INVERTED SCALE: Smooth areas (low std dev) become HIGH heat map values
            normalized_noise = min(noise_level / 50.0, 1.0) 
            heatmap_value = 1.0 - normalized_noise 
            
            heatmap[y:y+patch_h, x:x+patch_w] = heatmap_value
            
    return heatmap

# --- METADATA CHECK ---
def check_metadata(image_bytes):
    tags = exifread.process_file(io.BytesIO(image_bytes))
    editing_flags = []
    if 'Image Software' in tags:
        software = str(tags['Image Software'])
        if any(x in software.lower() for x in ['photoshop', 'lightroom', 'snapseed', 'picsart', 'facetune', 'midjourney', 'stable diffusion']):
            editing_flags.append(f"Software: {software}")
    return editing_flags if editing_flags else ["Clean Metadata"]

# --- UI MAIN ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# ALL IMAGE PROCESSING IS DONE INSIDE THIS IF BLOCK
if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    
    # Load image
    bytes_data = uploaded_file.getvalue()
    image = Image.open(uploaded_file).convert("RGB")
    
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.write(f"**Filename:** {uploaded_file.name}")
        
    # --- RUN ANALYSIS ---
    with st.spinner("Analyzing image layers..."):
        
        # 1. ELA Check
        ela_score, ela_vis = compute_ela(uploaded_file)
        if ela_score < 15:
            ela_verdict = "🔴 Heavy Editing / Compression Detected"
        elif 15 <= ela_score < 35:
            ela_verdict = "🟡 Moderate Edits Detected"
        else:
            ela_verdict = "🟢 Low Edits Detected (Likely original)"
            
        # 2. AI Check
        ai_prob = detect_ai_clip(image)
        if ai_prob > 60:
            ai_verdict = "🔴 AI GENERATED (High Probability)"
        elif 40 <= ai_prob <= 60:
            ai_verdict = "🟡 INCONCLUSIVE (Heavy filters or AI remixing)"
        else:
            ai_verdict = "🟢 REAL PHOTO (Low AI Probability)"
        
        # 3. AI Text Artifact Check (The Messi image killer)
        text_artifact_check = detect_ai_text_artifacts(image)
            
        # 4. Heatmap
        heatmap = generate_ai_smoothness_heatmap(image)

    # --- RESULTS DISPLAY ---
    with col2:
        st.subheader("🔬 Forensics Report")
        
        # Metadata
        meta = check_metadata(bytes_data)
        st.markdown(f"**Metadata:** `{', '.join(meta)}`")
        
        # AI Verdict
        st.markdown(f"**AI Probability:** `{ai_prob:.2f}%`")
        st.markdown(f"**Verdict:** {ai_verdict}")
        
        # Text Artifact Check Result
        st.markdown(f"**Text Analysis:** {text_artifact_check}")
        
        # Editing Verdict
        st.markdown(f"**ELA Score:** `{ela_score:.2f}`")
        st.markdown(f"**Editing Result:** {ela_verdict}")
        
        # --- HEATMAP VISUALIZATION ---
        st.subheader("🔥 Anomaly Heatmap (Unnaturally Smooth Regions)")
        st.write("Brighter yellow areas indicate unnaturally smooth textures. AI generated faces or objects often glow brightly here.")
        
        # Create composite visualization
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.imshow(image)
        ax.imshow(heatmap, cmap='inferno', alpha=0.5)
        ax.axis('off')
        st.pyplot(fig)
        
        st.info("**Note:** If the AI Probability is low but the heatmap shows bright yellow on the subject's face and clothing, it means the subject is AI-generated but pasted onto a real background.")