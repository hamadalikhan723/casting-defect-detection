"""Gradio app for casting defect detection."""
import logging
import os

import gradio as gr

from model import DefectClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger("defect-app")

classifier = DefectClassifier()


def predict(image, threshold):
    """Predict whether a casting image is normal or defective."""
    if image is None:
        raise gr.Error("Please upload an image.")

    try:
        result = classifier.predict(image, float(threshold))
    except ValueError as e:
        raise gr.Error(str(e))
    except Exception:
        log.exception("Prediction failed")
        raise gr.Error("Could not process this image. Please try another file.")

    log.info(
        "pred=%s conf=%.4f latency_ms=%.1f",
        result["predicted_class"],
        result["confidence"],
        result["latency_ms"],
    )

    scores = {
        "defective": result["p_defective"],
        "normal": 1 - result["p_defective"],
    }

    details = {
        "predicted_class": result["predicted_class"],
        "confidence": result["confidence"],
        "latency_ms": result["latency_ms"],
    }

    return scores, details


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Product image (JPEG/PNG)"),
        gr.Slider(
            0.05,
            0.95,
            value=0.5,
            step=0.05,
            label="Defect threshold "
                  "(lower = catch more defects, more false alarms)",
        ),
    ],
    outputs=[
        gr.Label(num_top_classes=2, label="Prediction"),
        gr.JSON(label="Details"),
    ],
    title="Casting Defect Detection",
    description=(
        "Upload a top-down photo of a cast pump impeller. "
        "The model (ResNet18, transfer learning) says whether it is "
        "normal or defective. Trained on the Casting Product dataset; "
        "it will not work on other products."
    ),
    api_name="predict",
    flagging_mode="never",
)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
    )