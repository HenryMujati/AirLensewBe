import streamlit as st
import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
import numpy as np
import folium
from streamlit_folium import st_folium
import io

# App title
st.set_page_config(page_title="Land Cover Prediction", layout="wide")
st.title("Land Cover Prediction with BigEarthNet Model")

# Model selection (default, can be changed)
MODEL_NAME = "mrm8488/convnext-tiny-finetuned-eurosat"

@st.cache_resource

def load_model():
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    return feature_extractor, model

feature_extractor, model = load_model()

# Image upload
uploaded_file = st.file_uploader(
    "Upload a satellite image (RGB, 224x224+ JPG/PNG/TIF)",
    type=["jpg", "jpeg", "png", "tif", "tiff"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits)[0].cpu().numpy()

    # Get class labels
    id2label = model.config.id2label
    results = [(id2label[i], float(prob)) for i, prob in enumerate(probs)]
    results = sorted(results, key=lambda x: x[1], reverse=True)

    # Show predictions (prob > 0.5)
    st.subheader("Predicted Land Cover Classes (p > 0.5)")
    for label, prob in results:
        if prob > 0.5:
            st.write(f"{label}: {prob:.2f}")

    # Show all probabilities as a table
    st.subheader("All Class Probabilities")
    st.dataframe({"Class": [r[0] for r in results], "Probability": [r[1] for r in results]})

    # Placeholder for graphs/heatmaps
    st.subheader("Prediction Graphs & Heatmaps (Coming Soon)")
    st.info("This section will show graphs and heatmaps of predictions.")

    # Save/delete buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save Results"):
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            st.download_button("Download Image", buf.getvalue(), file_name="prediction.png", mime="image/png")
    with col2:
        if st.button("Delete Results"):
            st.warning("Results deleted (refresh to upload a new image).")

    # Optional: Folium map
    st.subheader("Map Context (Optional)")
    m = folium.Map(location=[0, 0], zoom_start=2)
    st_folium(m, width=700, height=400)
else:
    st.info("Please upload a satellite image to get started.") 