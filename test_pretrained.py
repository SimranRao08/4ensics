from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch

MODEL_ID = "Ateeqq/ai-vs-human-image-detector"

print("Loading model (downloads ~400MB on first run, cached after)...")
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = SiglipForImageClassification.from_pretrained(MODEL_ID)
model.eval()

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits.softmax(dim=-1)[0]
    labels = model.config.id2label
    result = {labels[i]: float(probs[i]) for i in range(len(probs))}
    return result

# Test on your known images — update these filenames to match what's in your folder
test_images = [
    ("your_real_photo1.jpg", "REAL"),
    ("WhatsApp Image 2026-07-11 at 10.30.39 AM.jpeg", "FAKE"),
    ("WIN_20251111_11_19_17_Pro.jpg", "REAL"),
    ("filterman.jpeg", "REAL"),
    ("your_bike_ai.jpg", "FAKE"),
]

for path, expected in test_images:
    try:
        result = predict(path)
        print(f"{path} (expected {expected}): {result}")
    except FileNotFoundError:
        print(f"{path}: file not found, skipping")