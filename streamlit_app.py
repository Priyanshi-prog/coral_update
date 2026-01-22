import streamlit as st
import torch
import torch.nn as nn
import numpy as np
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

@st.cache_resource
def load_model():
    MODEL_NAME = "EPFL-ECEO/segformer-b2-finetuned-coralscapes-1024-1024"
    processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME)
    return processor, model

processor, model = load_model()

def run_image_analysis(image):
    image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        upsampled_logits = nn.functional.interpolate(
            outputs.logits, size=image.size[::-1], mode='bilinear'
        )
        mask = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()
    return mask

def get_prediction_image(mask):
    colors = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    # Define RGB for each category
    colors[np.isin(mask, HEALTHY_CORALS)] = [34, 139, 34]   # green
    colors[np.isin(mask, BLEACHED_CORALS)] = [255, 255, 255] # white
    colors[np.isin(mask, DEAD_CORALS)] = [169, 169, 169]     # grey
    colors[np.isin(mask, ALGAE_LIKE)] = [255, 0, 0]          # red
    return Image.fromarray(colors)

def generate_reef_report(mask, filename):
    total_px = mask.shape[0] * mask.shape[1]

    healthy_px = int(np.isin(mask, HEALTHY_CORALS).sum())
    bleached_px = int(np.isin(mask, BLEACHED_CORALS).sum())
    dead_px = int(np.isin(mask, DEAD_CORALS).sum())
    algae_px = int(np.isin(mask, ALGAE_LIKE).sum())

    live_coral_px = healthy_px + bleached_px

    # Percent cover metrics
    lcc_val = (live_coral_px / total_px) * 100
    sev_val = (bleached_px / live_coral_px * 100) if live_coral_px > 0 else 0
    algae_cov = (algae_px / total_px) * 100

    return {
        "Image_ID": str(filename),
        "Health_Status": "Bleached" if sev_val > 15 else "Healthy",
        "Bleaching_Severity_Score": round(float(sev_val), 2),
        "Severity_Label": "Severe" if sev_val > 50 else "Moderate" if sev_val > 15 else "Low",
        "Live_Coral_Cover_Pct": round(float(lcc_val), 1),
        "Algae_Cover_Pct": round(float(algae_cov), 1),
        "Dead_Coral_Pct": round(float((dead_px / total_px * 100)), 1)
    }

# --- 2. Define Real CoralScapes Class Groups ---

# Example coral class IDs from the Coralscapes dataset:
HEALTHY_CORALS = [6, 17, 31, 34, 36]    # other alive, massive alive, pocillopora alive, stylophora alive, meandering alive
BLEACHED_CORALS = [4, 16, 19, 33]       # other bleached, massive bleached, branching bleached, meandering bleached
DEAD_CORALS = [3, 20, 37, 32]           # other dead, branching dead, meandering dead, table acropora dead

# If you want to treat sand / substrate as "algae-like" for cover purposes:
ALGAE_LIKE = [5, 18]  # e.g., sand, rubble as proxy for non-coral benthic cover

# --- 3. STREAMLIT UI ---

st.title("🪸 Coral Reef Bleaching Detector")
st.markdown("### AI-Powered Analysis for Marine Conservation")

uploaded_file = st.file_uploader("Choose a Coral Image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    with st.spinner('Analyzing...'):
        image = Image.open(uploaded_file)
        mask = run_image_analysis(image)
        report = generate_reef_report(mask, uploaded_file.name)
        pred_img = get_prediction_image(mask)

    col1, col2, col3 = st.columns(3)
    col1.metric("Health Status", report["Health_Status"])
    col2.metric("Bleaching Severity", f"{report['Bleaching_Severity_Score']}%", report["Severity_Label"])
    col3.metric("Live Coral Cover", f"{report['Live_Coral_Cover_Pct']}%")

    st.subheader("Visual Output")
    st.image(image, caption="Original", use_container_width=True)
    st.image(pred_img, caption="Segmented Mask", use_container_width=True)

    st.subheader("Report")
    st.table([report])
