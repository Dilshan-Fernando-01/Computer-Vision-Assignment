import os
import sys

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.explainability.gradcam import generate_gradcam_overlay
from src.models.build_model import build_model
from src.preprocessing.preprocess import preprocess_image
from src.training.train import get_device

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "checkpoints", "variant_a_efficientnet_b0.pt")
EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "examples")
IMAGE_SIZE = 512
LOW_CONFIDENCE_THRESHOLD = 0.5

STAGE_NAMES = {
    0: "Stage 0 - No DR",
    1: "Stage 1 - Mild",
    2: "Stage 2 - Moderate",
    3: "Stage 3 - Severe",
    4: "Stage 4 - Proliferative",
}

DEVICE = get_device()
MODEL = build_model("efficientnet_b0", num_classes=5, pretrained=False)
MODEL.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
MODEL.to(DEVICE)
MODEL.eval()
TARGET_LAYER = MODEL.conv_head


def predict(image_rgb: np.ndarray):
    if image_rgb is None:
        return None, None, "Upload a retinal fundus photo to get a prediction."

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    preprocessed_bgr = preprocess_image(image_bgr, size=IMAGE_SIZE)

    rgb_float = cv2.cvtColor(preprocessed_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    input_tensor = torch.from_numpy(rgb_float).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = MODEL(input_tensor)
        probabilities = F.softmax(logits, dim=1)[0].cpu().numpy()

    predicted_class = int(probabilities.argmax())
    confidence = float(probabilities[predicted_class])

    label_scores = {STAGE_NAMES[i]: float(probabilities[i]) for i in range(5)}

    overlay = generate_gradcam_overlay(MODEL, TARGET_LAYER, input_tensor, rgb_float, target_class=predicted_class)

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        message = (
            f"**Low confidence ({confidence:.0%})** - this prediction is uncertain. "
            "Recommend review by a specialist rather than relying on this result alone."
        )
    else:
        message = f"Confidence: {confidence:.0%}"

    return label_scores, overlay, message


with gr.Blocks(title="Diabetic Retinopathy Stage Detection") as demo:
    gr.Markdown(
        "# Diabetic Retinopathy Stage Detection\n"
        "Upload a retinal fundus photo to get a predicted DR severity stage (ICDR 0-4 scale), "
        "a confidence score, and a Grad-CAM heatmap showing which regions drove the prediction.\n\n"
        "**Not a diagnostic tool** - a coursework research project, for demonstration only."
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="numpy", label="Retinal fundus photo")
            submit_btn = gr.Button("Predict", variant="primary")
            gr.Examples(
                examples=[os.path.join(EXAMPLES_DIR, f) for f in sorted(os.listdir(EXAMPLES_DIR))]
                if os.path.isdir(EXAMPLES_DIR)
                else [],
                inputs=image_input,
                label="Example fundus photos (one per stage, from the DDR test set)",
            )
        with gr.Column():
            label_output = gr.Label(label="Predicted stage probabilities")
            confidence_output = gr.Markdown()
            gradcam_output = gr.Image(label="Grad-CAM - regions that drove the prediction")

    submit_btn.click(fn=predict, inputs=image_input, outputs=[label_output, gradcam_output, confidence_output])
    image_input.change(fn=predict, inputs=image_input, outputs=[label_output, gradcam_output, confidence_output])


if __name__ == "__main__":
    demo.launch()
