import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from fpdf import FPDF
from datetime import date

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

# --- GENTLE IMAGE ENHANCER ---
def enhance_coral_image(pil_image):
    img = np.array(pil_image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Gentle Auto-White Balance
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    
    # Gentle Contrast
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.1, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l,a,b))
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Denoising
    result = cv2.fastNlMeansDenoisingColored(result, None, 3, 3, 7, 21)
    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

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
    total_pixels = mask.size
    coral_classes = [0,1,2,3,4,5,6,7,8,9,10,11,12,23,33,13,14,15,16,17,18]
    coral_pixels = np.isin(mask, coral_classes).sum()
    coverage_pct = (coral_pixels / total_pixels) * 100
    if coverage_pct < 2.0:
        return False, coverage_pct
    return True, coverage_pct

# --- NEW: GROWTH FORM CLASSIFIER ---
def classify_growth_form(mask):
    # create a binary mask of JUST the coral (healthy + bleached)
    coral_binary = np.isin(mask, [0,1,2,3,4,5,6,7,8,9,10,11,12,23,33]).astype(np.uint8) * 255
    
    # Find contours
    contours, _ = cv2.findContours(coral_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return "Unknown", 0
        
    # Get the largest coral chunk
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    
    if area < 100: return "Fragment", 0
    
    # Calculate Convex Hull (The "wrapper" around the shape)
    hull = cv2.convexHull(largest_contour)
    hull_area = cv2.contourArea(hull)
    
    # Solidity = How much "empty space" is inside the shape?
    # High Solidity (approx 1.0) = Solid/Round (Brain Coral)
    # Low Solidity (< 0.7) = Spiky/Branching (Staghorn)
    solidity = float(area) / hull_area if hull_area > 0 else 0
    
    if solidity > 0.85:
        return "Massive / Boulder (e.g., Porites)", solidity
    elif solidity < 0.65:
        return "Branching (e.g., Acropora)", solidity
    else:
        return "Plate / Encrusting", solidity

def get_prediction_image(mask):
    colors = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    colors[np.isin(mask, [0,1,2,3,4,5,6,7,8,9,10,11,23,33])] = [34, 139, 34] 
    colors[mask == 12] = [255, 255, 255] 
    colors[np.isin(mask, [13,14,15,16])] = [255, 0, 0] 
    colors[np.isin(mask, [17,18])] = [128, 128, 128] 
    return Image.fromarray(colors)

def plot_3d_structure(pil_image):
    img_small = pil_image.resize((150, 150)) 
    img_gray = np.array(img_small.convert('L'))
    fig = go.Figure(data=[go.Surface(z=img_gray, colorscale='Viridis')])
    fig.update_layout(
        title='3D Reef Topography (Rugosity)',
        autosize=True,
        width=500, height=500,
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z')
    )
    return fig

def generate_reef_report(mask, filename, metadata, growth_form):
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
        "Site_Name": metadata.get("site", "Unknown"),
        "Depth": metadata.get("depth", "Unknown"),
        "Date": metadata.get("date", str(date.today())),
        "Growth_Form": growth_form,
        "Health_Status": "Bleached" if sev_val > 10 else "Healthy",
        "Bleaching_Severity_Score": round(float(sev_val), 2),
        "Severity_Label": "Severe" if sev_val > 50 else "Moderate" if sev_val > 15 else "Low",
        "Live_Coral_Cover_Pct": round(float(lcc_val), 1),
        "Algae_Cover_Pct": round(float(algae_cov), 1),
        "Structural_Damage": "High" if damage_px > (total_px * 0.05) else "Low"
    }

def create_pdf(report):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Coral Reef Health Field Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Field Metadata:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Site: {report['Site_Name']} | Depth: {report['Depth']} | Date: {report['Date']}", ln=True)
    pdf.cell(200, 10, txt=f"Image Filename: {report['Image_ID']}", ln=True)
    pdf.cell(200, 10, txt=f"Identified Morphology: {report['Growth_Form']}", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt=f"Overall Status: {report['Health_Status']}", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.ln(5)
    pdf.cell(200, 10, txt=f"Bleaching Severity: {report['Severity_Label']} ({report['Bleaching_Severity_Score']}%)", ln=True)
    pdf.cell(200, 10, txt=f"Live Coral Cover: {report['Live_Coral_Cover_Pct']}%", ln=True)
    pdf.cell(200, 10, txt=f"Algae Cover: {report['Algae_Cover_Pct']}%", ln=True)
    pdf.ln(20)
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(200, 10, txt="Generated by AI Coral Inspector", ln=True, align='C')
    return pdf.output(dest='S').encode('latin-1')

# --- 3. FRONTEND UI ---
st.title("🪸 Coral Reef Bleaching Detector")
st.markdown("### AI-Powered Analysis for Marine Conservation")

st.sidebar.header("📝 Field Metadata")
site_name = st.sidebar.text_input("Survey Site Name", value="Site A")
depth_val = st.sidebar.text_input("Depth (meters)", value="5m")
survey_date = st.sidebar.date_input("Survey Date", date.today())
metadata = {"site": site_name, "depth": depth_val, "date": str(survey_date)}

uploaded_file = st.file_uploader("Choose a Coral Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    original_image = Image.open(uploaded_file)
    if original_image.width > 1024:
        new_height = int((1024 / original_image.width) * original_image.height)
        original_image = original_image.resize((1024, new_height))
    
    with st.spinner('Enhancing Image Quality...'):
        enhanced_image = enhance_coral_image(original_image)
    
    with st.spinner('Analyzing Coral Health...'):
        mask = run_image_analysis(original_image)
        is_coral, confidence = check_if_coral_present(mask)
        
        if not is_coral:
            st.error(f"⚠️ **Analysis Stopped:** No significant coral reef structures detected.")
            st.image(original_image, caption="Uploaded Image", width=300)
        else:
            # CLASSIFY SHAPE
            growth_form, solidity_score = classify_growth_form(mask)
            report = generate_reef_report(mask, uploaded_file.name, metadata, growth_form)
            prediction_img = get_prediction_image(mask)
            
            # --- TABS ---
            tab1, tab2, tab3 = st.tabs(["🔍 Analysis", "📊 Dashboard", "🏔️ 3D Structure"])
            
            with tab1:
                # NEW METRIC: MORPHOLOGY
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Health Status", report["Health_Status"])
                col2.metric("Bleaching", f"{report['Bleaching_Severity_Score']}%", report["Severity_Label"])
                col3.metric("Live Cover", f"{report['Live_Coral_Cover_Pct']}%")
                col4.metric("Morphology", growth_form, f"Solidity: {solidity_score:.2f}")
                
                img_col1, img_col2, img_col3 = st.columns(3)
                with img_col1: st.image(original_image, caption="Original", use_container_width=True)
                with img_col2: st.image(enhanced_image, caption="Enhanced View", use_container_width=True)
                with img_col3: st.image(prediction_img, caption="AI Segmentation Mask", use_container_width=True)
                st.caption("🟢 Green: Healthy | ⚪ White: Bleached | 🔴 Red: Algae | ⚫ Grey: Rubble")
            
            with tab2:
                chart_col, data_col = st.columns([1, 2])
                with chart_col:
                    labels = ['Healthy', 'Bleached', 'Algae', 'Rubble']
                    sizes = [report['Live_Coral_Cover_Pct'], report['Bleaching_Severity_Score'], report['Algae_Cover_Pct'], 100-(report['Live_Coral_Cover_Pct']+report['Bleaching_Severity_Score']+report['Algae_Cover_Pct'])]
                    colors_pie = ['#228B22', '#F0F8FF', '#FF0000', '#808080']
                    fig, ax = plt.subplots(figsize=(2, 2))
                    ax.pie(sizes, labels=None, autopct='%1.1f%%', startangle=90, colors=colors_pie, textprops={'fontsize': 6})
                    ax.axis('equal') 
                    st.pyplot(fig, use_container_width=False)
                
                with data_col:
                    st.write("**Field Data Log:**")
                    safe_report = {k: str(v) for k, v in report.items()}
                    st.table([safe_report])
                    
                    btn_col1, btn_col2 = st.columns(2)
                    with btn_col1:
                        pdf_bytes = create_pdf(report)
                        st.download_button("📄 Download PDF", pdf_bytes, f"reef_report.pdf", "application/pdf")
                    with btn_col2:
                        df = pd.DataFrame([safe_report])
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button("💾 Export CSV", csv, f"reef_data.csv", "text/csv")
            
            with tab3:
                st.subheader("3D Topographic Analysis (Rugosity)")
                fig_3d = plot_3d_structure(enhanced_image)
                st.plotly_chart(fig_3d, use_container_width=True)
            
            st.success(f"Analysis complete for {uploaded_file.name}")
