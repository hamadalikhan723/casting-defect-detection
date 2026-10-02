"""Model loading and inference for casting defect detection."""
import logging
import os
import time

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms as T

log = logging.getLogger("defect-model")

MODEL_PATH = os.getenv("MODEL_PATH", "resnet18_best.pt")
CLASSES = ["normal", "defective"]  # index 0 = normal, index 1 = defective
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class DefectClassifier:
    """ResNet18 (transfer learning) binary classifier: normal vs defective."""

    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}. "
                "Upload resnet18_best.pt to the root of the Space."
            )
        self.device = torch.device("cpu")
        self.transform = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])
        self.model = models.resnet18()
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        log.info("Model loaded from %s", model_path)

    def predict(self, image: Image.Image, threshold: float = 0.5) -> dict:
        """Return predicted class, confidence and latency for one PIL image.

        threshold: cutoff on P(defective). Lower it to catch more defects
        (higher recall) at the cost of more false alarms.
        """
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be between 0 and 1")

        image = image.convert("RGB")
        start = time.perf_counter()
        x = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            p_defective = torch.softmax(self.model(x), dim=1)[0, 1].item()
        latency_ms = (time.perf_counter() - start) * 1000

        label = "defective" if p_defective >= threshold else "normal"
        confidence = p_defective if label == "defective" else 1 - p_defective
        return {
            "predicted_class": label,
            "confidence": round(confidence, 4),
            "p_defective": round(p_defective, 4),
            "latency_ms": round(latency_ms, 1),
        }
