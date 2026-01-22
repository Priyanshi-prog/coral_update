import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageFilter
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from fpdf import FPDF
import io
import gc  # Garbage Collector to save RAM

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="ReefGuard AI",
    page_icon="🪸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Sea Glass" Design
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to bottom right, #e0f7fa, #ffffff);
    }
    div[data-testid="metric-container"] {
        background-color: rgba(255, 255, 255, 0.8);
        border: 1px solid #b2ebf2;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. BACKEND LOGIC ---
@st.cache_resource
def load_model():
    # Downloads model once and keeps it in memory
    MODEL_NAME = "EPFL-ECEO/segformer-b2-finetuned-coralscapes-1024-1024"
    processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME)
    return processor, model

processor, model = load_model()

def run_image_analysis(image):
    # MEMORY SAFEGUARD: Resize strictly to 1024px
    # This prevents the 1GB RAM limit from being hit
    input_image = image.resize((1024, 1024), Image.Resampling.LANCZOS)
    
    inputs = processor(images=input_image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        upsampled_logits = nn.functional.interpolate(
            outputs.logits, size=(1024, 1024), mode='bilinear'
        )
        mask = upsampled_logits.argmax(dim=1).squeeze().numpy()
    
    # FREE MEMORY IMMEDIATELY
    del inputs, outputs, upsampled_logits
    gc.collect()
        
    return mask

def clean_mask(mask):
    # Remove noise using standard PIL filter
    mask_img = Image.fromarray(mask.astype('uint8'))
    clean_img = mask_img.filter(ImageFilter.MedianFilter(size=5))
    return np.array(clean_img)

def get_prediction_image(mask):
    colors = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    # Healthy (Teal)
    colors[np.isin(mask, [0,1,2,3,4,5,6,7,8,9,10,11,23,33])] = [26, 188, 156]
    # Bleached (White)
    colors[mask == 12] = [236, 240, 241]
    # Algae (Red)
    colors[np.isin(mask, [13,14,15,16])] = [231, 76, 60]
    # Rubble (Grey)
    colors[np.isin(mask, [17,18])] = [149, 165, 166]
    return Image.fromarray(colors)

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
    
    status = "Healthy"
    if sev_val > 10: status = "Bleached"
    if sev_val > 50: status = "Critical"
    
    return {
        "File": str(filename),
        "Status": status,
        "Bleaching": round(float(sev_val), 2),
        "Live_Cover": round(float(lcc_val), 1),
        "Algae": round(float(algae_cov), 1),
    }

def create_pdf(report):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(0, 15, "ReefGuard AI Analysis", ln=True, align='C')
    
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(0, 10, f"File: {report['File']}", ln=True)
    pdf.cell(0, 10, f"Overall Status: {report['Status']}", ln=True)
    pdf.ln(5)
    
    pdf.cell(0, 10, f"Bleaching Severity: {report['Bleaching']}%", ln=True)
    pdf.cell(0, 10, f"Live Coral Cover: {report['Live_Cover']}%", ln=True)
    pdf.cell(0, 10, f"Algae Coverage: {report['Algae']}%", ln=True)
    
    return pdf.output(dest='S').encode('latin-1')

# --- 3. FRONTEND UI ---
st.title("🪸 ReefGuard AI")
st.markdown("### Intelligent Coral Health Monitoring System")

uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Open Image
    image = Image.open(uploaded_file)
    
    # 2. Safety Resize (Display Version)
    if image.width > 1024:
        image.thumbnail((1024, 1024))
        
    with st.spinner('Processing reef data...'):
        # 3. Run Analysis
        raw_mask = run_image_analysis(image)
        clean_mask = clean_mask(raw_mask)
        report = generate_report(clean_mask, uploaded_file.name)
        prediction = get_prediction_image(clean_mask)
        
        # 4. Cleanup RAM
        gc.collect()

    # Dashboard
    st.markdown("#### Real-time Telemetry")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Health Status", report["Status"])
    m2.metric("Bleaching", f"{report['Bleaching']}%")
    m3.metric("Live Cover", f"{report['Live_Cover']}%")
    m4.metric("Algae", f"{report['Algae']}%")
    
    st.markdown("---")

    # Visuals
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### 📷 Original Feed")
        st.image(image, use_container_width=True)
    with c2:
        st.markdown("##### 🧠 AI Diagnostics")
        st.image(prediction, use_container_width=True)
        
    st.caption("Legend: 🟢 Healthy (Green) | ⚪ Bleached (White) | 🔴 Algae (Red) | ⚫ Rubble (Grey)")

    # Export
    st.markdown("---")
    pdf_data = create_pdf(report)
    st.download_button(
        label="📄 Export Scientific Report (PDF)",
        data=pdf_data,
        file_name=f"ReefGuard_Report.pdf",
        mime="application/pdf",
        type="primary"
    )

else:
    st.info("Please upload an image from the sidebar to begin analysis.")
