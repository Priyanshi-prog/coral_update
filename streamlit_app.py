import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageFilter # Using native PIL for filtering
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from fpdf import FPDF

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Coral Reef AI",
    page_icon="🪸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. BACKEND LOGIC ---
@st.cache_resource
def load_model():
    # Downloads model automatically from Hugging Face
    MODEL_NAME = "EPFL-ECEO/segformer-b2-finetuned-coralscapes-1024-1024"
    processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME)
    return processor, model

processor, model = load_model()

def run_image_analysis(image):
    # CRITICAL: Resize to 1024x1024 immediately to save RAM.
    # LANCZOS is the highest quality resize method.
    input_image = image.resize((1024, 1024), Image.Resampling.LANCZOS)
    
    inputs = processor(images=input_image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        upsampled_logits = nn.functional.interpolate(
            outputs.logits, size=(1024, 1024), mode='bilinear'
        )
        mask = upsampled_logits.argmax(dim=1).squeeze().numpy()
        
    return mask

def get_prediction_image(mask):
    # Convert mask to colored image
    colors = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    colors[np.isin(mask, [0,1,2,3,4,5,6,7,8,9,10,11,23,33])] = [34, 139, 34] # Green
    colors[mask == 12] = [255, 255, 255] # White
    colors[np.isin(mask, [13,14,15,16])] = [220, 20, 60] # Red
    colors[np.isin(mask, [17,18])] = [128, 128, 128] # Grey
    
    img = Image.fromarray(colors)
    
    # LIGHTWEIGHT DENOISING: Use PIL's built-in MedianFilter
    # This removes the need for the heavy 'scipy' library
    img = img.filter(ImageFilter.MedianFilter(size=5))
    return img

def generate_reef_report(mask, filename):
    total_px = int(mask.size)
    if total_px == 0: total_px = 1
    
    healthy_px = int(np.isin(mask, [0,1,2,3,4,5,6,7,8,9,10,11,23,33]).sum())
    bleached_px = int((mask == 12).sum())
    algae_px = int(np.isin(mask, [13,14,15,16]).sum())
    
    total_coral = healthy_px + bleached_px
    
    lcc_val = (total_coral / total_px) * 100
    sev_val = (bleached_px / total_coral * 100) if total_coral > 0 else 0
    algae_cov = (algae_px / total_px) * 100
    
    return {
        "Image_ID": str(filename),
        "Health_Status": "Bleached" if sev_val > 10 else "Healthy",
        "Bleaching_Severity": round(float(sev_val), 2),
        "Live_Coral_Cover": round(float(lcc_val), 1),
        "Algae_Cover": round(float(algae_cov), 1),
    }

def create_pdf(report):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Coral Reef Health Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"File: {report['Image_ID']}", ln=True)
    pdf.ln(5)
    
    status_color = "RED" if report['Health_Status'] == "Bleached" else "GREEN"
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt=f"Status: {report['Health_Status']}", ln=True)
    
    pdf.set_font("Arial", size=12)
    pdf.ln(5)
    pdf.cell(200, 10, txt=f"Bleaching Severity: {report['Bleaching_Severity']}%", ln=True)
    pdf.cell(200, 10, txt=f"Live Coral Cover: {report['Live_Coral_Cover']}%", ln=True)
    pdf.cell(200, 10, txt=f"Algae Cover: {report['Algae_Cover']}%", ln=True)
    
    return pdf.output(dest='S').encode('latin-1')

# --- 3. FRONTEND UI ---
st.title("🪸 Coral Reef AI Inspector")
st.markdown("### Automated Health Analysis System")

uploaded_file = st.file_uploader("Upload Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Open & Resize Immediately
    original_image = Image.open(uploaded_file)
    if original_image.width > 1024:
        original_image.thumbnail((1024, 1024))

    with st.spinner('Analyzing...'):
        # 2. Run AI Analysis
        raw_mask = run_image_analysis(original_image)
        
        # 3. Generate Outputs
        report = generate_reef_report(raw_mask, uploaded_file.name)
        prediction_img = get_prediction_image(raw_mask)

    # --- UI DISPLAY ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Status", report["Health_Status"])
    col2.metric("Bleaching", f"{report['Bleaching_Severity']}%")
    col3.metric("Live Cover", f"{report['Live_Coral_Cover']}%")
    col4.metric("Algae", f"{report['Algae_Cover']}%")

    st.divider()

    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.image(original_image, caption="Original Input", use_container_width=True)
    with col_img2:
        st.image(prediction_img, caption="AI Segmentation Mask", use_container_width=True)
    
    st.caption("Legend: 🟢 Healthy | ⚪ Bleached | 🔴 Algae | ⚫ Rubble")
    
    st.divider()

    col_btn = st.columns(1)[0]
    with col_btn:
        pdf_bytes = create_pdf(report)
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_bytes,
            file_name=f"reef_report.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )
