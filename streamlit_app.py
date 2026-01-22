import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageFilter
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from fpdf import FPDF
import io

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
    /* Main Background - Subtle Ocean Gradient */
    .stApp {
        background: linear-gradient(to bottom right, #e0f7fa, #ffffff);
    }
    
    /* Card Styling */
    div[data-testid="metric-container"] {
        background-color: rgba(255, 255, 255, 0.8);
        border: 1px solid #b2ebf2;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.1);
    }
    
    /* Header Styling */
    h1 {
        color: #006064;
        font-family: 'Helvetica Neue', sans-serif;
    }
    h3 {
        color: #00838f;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #f0fcfc;
        border-right: 1px solid #b2ebf2;
    }
    
    /* Button Styling */
    div.stButton > button {
        background: linear-gradient(45deg, #00acc1, #00838f);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
    }
    div.stButton > button:hover {
        background: linear-gradient(45deg, #00bcd4, #0097a7);
        border: none;
        box-shadow: 0 4px 12px rgba(0,172,193,0.3);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. BACKEND LOGIC (Lightweight & Stable) ---
@st.cache_resource
def load_model():
    MODEL_NAME = "EPFL-ECEO/segformer-b2-finetuned-coralscapes-1024-1024"
    processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME)
    return processor, model

processor, model = load_model()

def run_image_analysis(image):
    # Resize to 1024x1024 to prevent RAM crash
    input_image = image.resize((1024, 1024), Image.Resampling.LANCZOS)
    inputs = processor(images=input_image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        upsampled_logits = nn.functional.interpolate(
            outputs.logits, size=(1024, 1024), mode='bilinear'
        )
        mask = upsampled_logits.argmax(dim=1).squeeze().numpy()
        
    return mask

def clean_mask(mask):
    # PIL Median Filter (Lightweight Despeckling)
    mask_img = Image.fromarray(mask.astype('uint8'))
    clean_img = mask_img.filter(ImageFilter.MedianFilter(size=5))
    return np.array(clean_img)

def get_prediction_image(mask):
    colors = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    # Healthy (Teal/Green)
    colors[np.isin(mask, [0,1,2,3,4,5,6,7,8,9,10,11,23,33])] = [26, 188, 156]
    # Bleached (White)
    colors[mask == 12] = [236, 240, 241]
    # Algae (Coral Red)
    colors[np.isin(mask, [13,14,15,16])] = [231, 76, 60]
    # Rubble (Slate Grey)
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
    pdf.set_font("Arial", 'I', 12)
    pdf.cell(0, 10, "Automated Ecological Assessment", ln=True, align='C')
    pdf.ln(10)
    
    # Metrics Box
    pdf.set_fill_color(240, 252, 252) # Light Cyan
    pdf.rect(10, 40, 190, 60, 'F')
    
    pdf.set_font("Arial", 'B', 14)
    pdf.set_xy(20, 50)
    pdf.cell(0, 10, f"File: {report['File']}", ln=True)
    pdf.set_xy(20, 60)
    pdf.cell(0, 10, f"Overall Status: {report['Status']}", ln=True)
    
    pdf.set_font("Arial", size=12)
    pdf.set_xy(20, 75)
    pdf.cell(0, 10, f"Bleaching Severity: {report['Bleaching']}%", ln=True)
    pdf.set_xy(20, 85)
    pdf.cell(0, 10, f"Live Coral Cover: {report['Live_Cover']}%", ln=True)
    pdf.set_xy(20, 95)
    pdf.cell(0, 10, f"Algae Coverage: {report['Algae']}%", ln=True)
    
    pdf.set_y(110)
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 10, "Report generated by ReefGuard AI System", align='C')
    return pdf.output(dest='S').encode('latin-1')

# --- 3. FRONTEND UI ---
st.title("🪸 ReefGuard AI")
st.markdown("### Intelligent Coral Health Monitoring System")

# Sidebar
st.sidebar.title("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload Underwater Image", type=["jpg", "jpeg", "png"])
st.sidebar.markdown("---")
st.sidebar.info("This system uses the EPFL SegFormer model to detect bleaching events with pixel-level precision.")

if uploaded_file is not None:
    # Processing
    image = Image.open(uploaded_file)
    if image.width > 1024:
        image.thumbnail((1024, 1024))
        
    with st.spinner('Processing reef data...'):
        raw_mask = run_image_analysis(image)
        clean_mask = clean_mask(raw_mask)
        report = generate_report(clean_mask, uploaded_file.name)
        prediction = get_prediction_image(clean_mask)

    # Metrics Dashboard
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
        st.image(image, use_container_width=True, caption="Input Image")
    with c2:
        st.markdown("##### 🧠 AI Diagnostics")
        st.image(prediction, use_container_width=True, caption="Segmentation Mask")
        
    st.info("Legend: 🟢 Healthy (Green) | ⚪ Bleached (White) | 🔴 Algae (Red) | ⚫ Rubble (Grey)")

    # Export
    st.markdown("---")
    col_dl, col_space = st.columns([1, 2])
    with col_dl:
        pdf_data = create_pdf(report)
        st.download_button(
            label="📄 Export Scientific Report (PDF)",
            data=pdf_data,
            file_name=f"ReefGuard_Report.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )

else:
    # Welcome Screen
    st.markdown("""
    <div style='background-color: #f0fcfc; padding: 20px; border-radius: 10px; border: 1px solid #b2ebf2;'>
        <h4>Welcome to ReefGuard</h4>
        <p>Upload a coral reef image to begin analysis. The system will detect:</p>
        <ul>
            <li>Live Coral Coverage</li>
            <li>Bleaching Severity</li>
            <li>Algae Proliferation</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
