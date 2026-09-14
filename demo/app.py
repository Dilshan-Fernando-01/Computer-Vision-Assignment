"""Gradio UI - for local testing. The live deployment uses demo/streamlit_app.py
on Streamlit Community Cloud instead (Hugging Face Spaces' free tier no longer
supports Gradio without a paid plan - see Temp/DEVELOPMENT_LOG.md).
"""

import os

import gradio as gr

from predictor import predict

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "examples")


def run(image_rgb):
    if image_rgb is None:
        return None, None, "Upload a retinal fundus photo to get a prediction."
    return predict(image_rgb)


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

    submit_btn.click(fn=run, inputs=image_input, outputs=[label_output, gradcam_output, confidence_output])
    image_input.change(fn=run, inputs=image_input, outputs=[label_output, gradcam_output, confidence_output])


if __name__ == "__main__":
    demo.launch()
