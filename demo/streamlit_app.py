import os

import streamlit as st

from predictor import predict

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "examples")

st.set_page_config(page_title="Diabetic Retinopathy Stage Detection", layout="wide")

st.title("Diabetic Retinopathy Stage Detection")
st.markdown(
    "Upload a retinal fundus photo to get a predicted DR severity stage (ICDR 0-4 scale), "
    "a confidence score, and a Grad-CAM heatmap showing which regions drove the prediction.\n\n"
    "**Not a diagnostic tool** - a coursework research project, for demonstration only."
)

left, right = st.columns(2)

with left:
    uploaded_file = st.file_uploader("Retinal fundus photo", type=["jpg", "jpeg", "png"])

    st.caption("Or try an example (one per stage, from the DDR test set):")
    example_files = sorted(os.listdir(EXAMPLES_DIR)) if os.path.isdir(EXAMPLES_DIR) else []
    example_cols = st.columns(len(example_files)) if example_files else []
    selected_example = None
    for col, filename in zip(example_cols, example_files):
        with col:
            st.image(os.path.join(EXAMPLES_DIR, filename), use_container_width=True)
            if st.button("Use this", key=filename):
                selected_example = os.path.join(EXAMPLES_DIR, filename)

image_path = None
if uploaded_file is not None:
    image_path = uploaded_file
elif selected_example is not None:
    image_path = selected_example

if image_path is not None:
    import numpy as np
    from PIL import Image

    image_rgb = np.array(Image.open(image_path).convert("RGB"))

    with left:
        st.image(image_rgb, caption="Input photo", use_container_width=True)

    label_scores, overlay, message = predict(image_rgb)

    with right:
        st.subheader("Predicted stage probabilities")
        st.bar_chart(label_scores)
        st.markdown(message)
        st.subheader("Grad-CAM - regions that drove the prediction")
        st.image(overlay, use_container_width=True)
else:
    with right:
        st.info("Upload a photo or pick an example to get a prediction.")
