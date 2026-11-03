"""
screenshot_forensics_final.py
-------------------------
Upgraded Screenshot / Chat Evidence Detector.
Fixes applied: 
1. Abandoned ELA and Z-Scores: They mathematically fail on WhatsApp-compressed images and images with very little text.
2. Color Patch Detection: Checks if the background color immediately behind a word was painted over with a brush.
3. Artificial Flatness Check: Checks if the background behind a word has an unnatural 0.0 noise variance (a solid drawn box).
"""

import cv2
import numpy as np

def _merge_overlapping_boxes(boxes, y_overlap_thresh=0.4, max_x_dist=40):
    """Groups individual words into single text lines."""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    merged = []
    used = [False] * len(boxes)

    for i, (x1, y1, w1, h1) in enumerate(boxes):
        if used[i]: continue
        group = [(x1, y1, w1, h1)]
        used[i] = True
        for j in range(i + 1, len(boxes)):
            if used[j]: continue
            x2, y2, w2, h2 = boxes[j]
            overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            dist_x = x2 - (x1 + w1)
            
            if overlap_y > y_overlap_thresh * min(h1, h2) and -10 <= dist_x <= max_x_dist:
                group.append((x2, y2, w2, h2))
                used[j] = True
                
        gx0 = min(b[0] for b in group)
        gy0 = min(b[1] for b in group)
        gx1 = max(b[0] + b[2] for b in group)
        gy1 = max(b[1] + b[3] for b in group)
        merged.append((gx0, gy0, gx1 - gx0, gy1 - gy0))
    return merged

def detect_text_line_regions(gray, min_area=30):
    """Finds bounding boxes for all text lines in the image."""
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 2))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h > min_area and 5 < h < gray.shape[0] * 0.15:
            boxes.append((x, y, w, h))

    return _merge_overlapping_boxes(boxes)

def detect_painted_background_patches(img_bgr, text_boxes, expand=15):
    """
    The Core Engine: Detects if someone used a brush tool to paint over an old number.
    It isolates the background inside the text box and compares it to the background outside.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    anomalies = []

    for (x, y, w, h) in text_boxes:
        # Define a slightly larger box to compare the inside vs the outside
        y0, y1 = max(0, y - expand), min(img_bgr.shape[0], y + h + expand)
        x0, x1 = max(0, x - expand), min(img_bgr.shape[1], x + w + expand)
        
        roi_bgr = img_bgr[y0:y1, x0:x1]
        roi_gray = gray[y0:y1, x0:x1]
        
        # 1. Mask out the actual text/letters using Edge Detection
        edges = cv2.Canny(roi_gray, 30, 100)
        text_mask = cv2.dilate(edges, np.ones((5, 5), np.uint8))
        bg_mask = cv2.bitwise_not(text_mask) # Keep ONLY the empty background
        
        # 2. Split the background into "Inside the Box" and "Outside the Box"
        inner_mask = np.zeros_like(bg_mask)
        inner_y, inner_x = y - y0, x - x0
        inner_mask[inner_y:inner_y+h, inner_x:inner_x+w] = 255
        
        inner_bg_mask = cv2.bitwise_and(bg_mask, inner_mask)
        outer_bg_mask = cv2.bitwise_and(bg_mask, cv2.bitwise_not(inner_mask))
        
        # Ensure we have enough background pixels to do math on
        if cv2.countNonZero(inner_bg_mask) < 20 or cv2.countNonZero(outer_bg_mask) < 20:
            continue
            
        # 3. METRIC A: Color Patching (Did they paint a box that doesn't match?)
        inner_color = cv2.mean(roi_bgr, mask=inner_bg_mask)[:3]
        outer_color = cv2.mean(roi_bgr, mask=outer_bg_mask)[:3]
        color_diff = np.linalg.norm(np.array(inner_color) - np.array(outer_color))
        
        # 4. METRIC B: Artificial Flatness (Is the drawn box unnaturally lacking JPEG noise?)
        _, inner_stddev = cv2.meanStdDev(roi_gray, mask=inner_bg_mask)
        _, outer_stddev = cv2.meanStdDev(roi_gray, mask=outer_bg_mask)
        inner_var = inner_stddev[0][0] ** 2
        outer_var = outer_stddev[0][0] ** 2
        
        # FLAG RULE: 
        # If the background color is off by > 6.0 RGB points OR 
        # the inside is completely flat (variance < 0.5) while the outside has normal noise (variance > 2.0)
        if color_diff > 6.0 or (inner_var < 0.5 and outer_var > 2.0):
            anomalies.append((x, y, w, h))
            
    return anomalies

def analyze_screenshot(image_path):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(image_path)
    
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    text_boxes = detect_text_line_regions(gray)
    
    patched_anomalies = detect_painted_background_patches(img_bgr, text_boxes)
    
    indicators = []
    score = 0.0

    if len(patched_anomalies) > 0:
        score += min(85, len(patched_anomalies) * 45)
        indicators.append(f"{len(patched_anomalies)} text region(s) show Background Color Mismatch or Artificial Flatness — indicating a painted-over edit.")

    score = min(100, score)

    if not indicators:
        indicators.append("No background patching detected. Image appears consistent.")

    if score >= 60:
        priority = "High"
    elif score >= 25:
        priority = "Medium"
    else:
        priority = "Low"

    return {
        "screenshot_risk_score": round(score, 1),
        "priority_level": priority,
        "forensic_indicators": indicators,
        "text_lines_detected": len(text_boxes),
        "patched_regions_detected": len(patched_anomalies)
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python screenshot_forensics_final.py <image_path>")
        sys.exit(1)

    result = analyze_screenshot(sys.argv[1])
    print(f"\nImage: {sys.argv[1]}")
    print(f"Screenshot Risk Score: {result['screenshot_risk_score']}/100")
    print(f"Priority: {result['priority_level']}")
    print(f"Text lines detected: {result['text_lines_detected']}")
    print(f"Painted-over Boxes Detected: {result['patched_regions_detected']}")
    print(f"\nIndicators:")
    for ind in result["forensic_indicators"]:
        print(f"  - {ind}")