import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageFilter
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from fpdf import FPDF
import io
import gc

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="ReefGuard AI",
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

def run_image_analysis(image):
    # CRITICAL STABILITY FIX: Resize to 512x512
    # 1024x1024 is too heavy for the free cloud tier and causes crashes.
    # 512x512 is 4x lighter and still accurate enough for a demo.
    image = image.convert("RGB")
    input_image = image.resize((512, 512))
    
    inputs = processor(images=input_image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        # Upsample to match the 512x512 input
        upsampled_logits = nn.functional.interpolate(
            outputs.logits, size=(512, 512), mode='bilinear'
        )
        mask = upsampled_logits.argmax(dim=1).squeeze().numpy()
    
    del inputs, outputs, upsampled_logits
    gc.collect()
    return mask

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
    
    img = Image.fromarray(colors)
    # Accuracy Boost: Clean noise dots
    img = img.filter(ImageFilter.MedianFilter(size=3))
    return img

def generate_report(mask, filename):
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
        "File": str(filename),
        "Status": "Bleached" if sev_val > 10 else "Healthy",
        "Bleaching": round(float(sev_val), 2),
        "Live_Cover": round(float(lcc_val), 1),
        "Algae": round(float(algae_cov), 1),
    }

def create_pdf(report):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Coral Health Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"File: {report['File']}", ln=True)
    pdf.cell(200, 10, txt=f"Status: {report['Status']}", ln=True)
    pdf.ln(5)
    pdf.cell(200, 10, txt=f"Bleaching: {report['Bleaching']}%", ln=True)
    pdf.cell(200, 10, txt=f"Live Cover: {report['Live_Cover']}%", ln=True)
    pdf.cell(200, 10, txt=f"Algae: {report['Algae']}%", ln=True)
    return pdf.output(dest='S').encode('latin-1')

# --- 3. FRONTEND UI ---
st.title("🪸 ReefGuard AI")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Resize for display to match analysis
    image = Image.open(uploaded_file).convert("RGB")
    display_image = image.resize((512, 512))
    
    with st.spinner('Analyzing...'):
        mask = run_image_analysis(image)
        report = generate_report(mask, uploaded_file.name)
        prediction = get_prediction_image(mask)
        gc.collect()

    col1, col2, col3 = st.columns(3)
    col1.metric("Status", report["Status"])
    col2.metric("Bleaching", f"{report['Bleaching']}%")
    col3.metric("Algae", f"{report['Algae']}%")

    c1, c2 = st.columns(2)
    with c1:
        st.image(display_image, caption="Original", use_container_width=True)
    with c2:
        st.image(prediction, caption="AI Analysis", use_container_width=True)
        
    st.caption("🟢 Green: Healthy | ⚪ White: Bleached | 🔴 Red: Algae")

    # Safe PDF Download
    pdf_bytes = create_pdf(report)
    st.download_button("Download PDF Report", pdf_bytes, "report.pdf", "application/pdf")
