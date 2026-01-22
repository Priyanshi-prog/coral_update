import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from fpdf import FPDF
import io

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Coral Reef AI Inspector",
    page_icon="🪸",
    layout="wide"
)

# --- 2. BACKEND LOGIC ---
@st.cache_resource
def load_model():
    MODEL_NAME = "EPFL-ECEO/segformer-b2-finetuned-coralscapes-1024-1024"
    processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME)
    return processor, model

processor, model = load_model()

# --- IMAGE ENHANCER ---
def enhance_coral_image(pil_image):
    # Convert PIL Image to OpenCV format (numpy array)
    img = np.array(pil_image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Convert RGB to BGR

    # 1. Color correction (reduce blue tint)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_lab)

    # Apply CLAHE to L channel for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)

    # Reduce blue channel (b channel in LAB)
    b = cv2.addWeighted(b, 0.7, np.ones_like(b)*128, 0.3, 0)

    # Merge back
    enhanced_lab = cv2.merge([l, a, b])
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # 2. Sharpen slightly
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    enhanced_bgr = cv2.filter2D(enhanced_bgr, -1, kernel)

    # Convert back to PIL Image (RGB) for Streamlit
    enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(enhanced_rgb)

def run_image_analysis(image):
    image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        upsampled_logits = nn.functional.interpolate(
            outputs.logits, size=image.size[::-1], mode='bilinear'
        )
        mask = upsampled_logits.argmax(dim=1).squeeze().numpy()
    return mask

def check_if_coral_present(mask):
    """
    Checks if the AI actually found coral in the image.
    Returns True if > 2% of the image is classified as coral.
    """
    total_pixels = mask.size
    # IDs for Healthy (0-11, 23, 33), Bleached (12), Dead (17-18), Algae (13-16)
    coral_classes = [0,1,2,3,4,5,6,7,8,9,10,11,12,23,33,13,14,15,16,17,18]
    coral_pixels = np.isin(mask, coral_classes).sum()
    
    coverage_pct = (coral_pixels / total_pixels) * 100
    
    # THRESHOLD: If less than 2% is coral, assume it's empty water/sand
    if coverage_pct < 2.0:
        return False, coverage_pct
    return True, coverage_pct

def get_prediction_image(mask):
    colors = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    # Healthy (Green)
    colors[np.isin(mask, [0,1,2,3,4,5,6,7,8,9,10,11,23,33])] = [34, 139, 34]
    # Bleached (White)
    colors[mask == 12] = [255, 255, 255]
    # Algae (Red)
    colors[np.isin(mask, [13,14,15,16])] = [255, 0, 0]
    # Rubble (Grey)
    colors[np.isin(mask, [17,18])] = [128, 128, 128]
    return Image.fromarray(colors)

def generate_reef_report(mask, filename):
    # Safety Cast: Convert numpy numbers to standard Python int/float
    total_px = int(mask.size)
    healthy_px = int(np.isin(mask, [0,1,2,3,4,5,6,7,8,9,10,11,23,33]).sum())
    bleached_px = int((mask == 12).sum())
    algae_px = int(np.isin(mask, [13,14,15,16]).sum())
    damage_px = int(np.isin(mask, [17,18]).sum())
    
    total_coral = healthy_px + bleached_px
    lcc_val = (total_coral / total_px) * 100
    sev_val = (bleached_px / total_coral * 100) if total_coral > 0 else 0
    algae_cov = (algae_px / total_px) * 100
    
    return {
        "Image_ID": str(filename),
        "Health_Status": "Bleached" if sev_val > 10 else "Healthy",
        "Bleaching_Severity_Score": round(float(sev_val), 2),
        "Severity_Label": "Severe" if sev_val > 50 else "Moderate" if sev_val > 15 else "Low",
        "Live_Coral_Cover_Pct": round(float(lcc_val), 1),
        "Algae_Cover_Pct": round(float(algae_cov), 1),
        "Structural_Damage": "High" if damage_px > (total_px * 0.05) else "Low"
    }

# --- PDF GENERATOR ---
def create_pdf(report):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Coral Reef Health Analysis Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Image Filename: {report['Image_ID']}", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt=f"Overall Status: {report['Health_Status']}", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.ln(5)
    pdf.cell(200, 10, txt=f"Bleaching Severity: {report['Severity_Label']} ({report['Bleaching_Severity_Score']}%)", ln=True)
    pdf.cell(200, 10, txt=f"Live Coral Cover: {report['Live_Coral_Cover_Pct']}%", ln=True)
    pdf.cell(200, 10, txt=f"Algae Cover: {report['Algae_Cover_Pct']}%", ln=True)
    pdf.cell(200, 10, txt=f"Structural Damage: {report['Structural_Damage']}", ln=True)
    pdf.ln(20)
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(200, 10, txt="Generated by AI Coral Inspector", ln=True, align='C')
    return pdf.output(dest='S').encode('latin-1')

# --- 3. FRONTEND UI ---
st.title("🪸 Coral Reef Bleaching Detector")
st.markdown("### AI-Powered Analysis for Marine Conservation")

uploaded_file = st.file_uploader("Choose a Coral Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # --- MEMORY FIX: Resize huge images immediately ---
    original_image = Image.open(uploaded_file)
    
    # Resize logic: If width > 1024px, shrink it to save RAM
    if original_image.width > 1024:
        new_height = int((1024 / original_image.width) * original_image.height)
        original_image = original_image.resize((1024, new_height))

    # 1. ENHANCEMENT STEP
    with st.spinner('Enhancing Image Quality...'):
        enhanced_image = enhance_coral_image(original_image)
    
    # 2. ANALYSIS STEP
    with st.spinner('Analyzing Coral Health...'):
        mask = run_image_analysis(enhanced_image)
        
        # 3. DETECTION CHECK (Is it a coral?)
        is_coral, confidence = check_if_coral_present(mask)

        if not is_coral:
            st.error(f"⚠️ **Analysis Stopped:** No significant coral reef structures detected.")
            st.warning(f"The model found only {confidence:.2f}% coral coverage. Please upload a clearer underwater image containing visible coral.")
            
            # Show images so user understands why
            img_col1, img_col2 = st.columns(2)
            with img_col1:
                st.image(original_image, caption="Uploaded Image", width=300)
            with img_col2:
                st.image(enhanced_image, caption="Enhanced Image (No Coral Found)", width=300)
                
        else:
            # PROCEED WITH REPORT
            report = generate_reef_report(mask, uploaded_file.name)
            prediction_img = get_prediction_image(mask)

            # --- DISPLAY RESULTS ---
            col1, col2, col3 = st.columns(3)
            col1.metric("Health Status", report["Health_Status"])
            col2.metric("Bleaching Severity", f"{report['Bleaching_Severity_Score']}%", report["Severity_Label"])
            col3.metric("Live Coral Cover", f"{report['Live_Coral_Cover_Pct']}%")

            st.subheader("Visual Analysis")
            img_col1, img_col2, img_col3 = st.columns(3)
            with img_col1:
                st.image(original_image, caption="Original", use_container_width=True)
            with img_col2:
                st.image(enhanced_image, caption="Enhanced (Color Corrected)", use_container_width=True)
            with img_col3:
                st.image(prediction_img, caption="AI Segmentation Mask", use_container_width=True)
            
            st.caption("🟢 Green: Healthy | ⚪ White: Bleached | 🔴 Red: Algae | ⚫ Grey: Rubble")

            st.subheader("Detailed Report")
            # Convert to strings to prevent Table Crash
            safe_report = {k: str(v) for k, v in report.items()}
            st.table([safe_report])
            
            pdf_bytes = create_pdf(report)
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_bytes,
                file_name=f"reef_report_{uploaded_file.name}.pdf",
                mime="application/pdf"
            )

            st.success(f"Analysis complete for {uploaded_file.name}")
