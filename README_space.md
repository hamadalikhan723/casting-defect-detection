---
title: Casting Defect Detection
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# Casting Defect Detection

Upload a photo of a cast pump impeller and the model tells you if it is **normal** or **defective**, with a confidence score.

- **Model:** ResNet18, ImageNet-pretrained, fine-tuned (transfer learning)
- **Data:** Casting Product Image Data (Kaggle), split by duplicate groups to avoid train/test leakage
- **Test result:** about 99.9% accuracy on 1,057 unseen images (replace with your final numbers)
- **Latency:** about 60-80 ms per image on CPU

## How to use

1. Upload an image (JPEG or PNG).
2. Optionally move the threshold slider. Lower values catch more defects but raise false alarms.
3. Read the predicted class and confidence.

## Use it as an API

```python
from gradio_client import Client, handle_file

client = Client("YOUR_USERNAME/YOUR_SPACE_NAME")
scores, details = client.predict(handle_file("image.jpg"), 0.5, api_name="/predict")
print(details)  # {'predicted_class': 'defective', 'confidence': 0.99, 'latency_ms': 70.0}
```

## Files

- `app.py`: Gradio interface, input validation, logging, error handling
- `model.py`: model loading and inference
- `resnet18_best.pt`: trained weights
- `requirements.txt`: dependencies

## Known limitations

- Trained on one part type and one camera angle. It will not work on other products.
- Defective parts are slightly darker on average (brightness alone gives 83% accuracy), so the model may partly rely on lighting.
- Confidence is an uncalibrated softmax score, not a true probability.
- Any image, even an unrelated one, gets labeled normal or defective.
- Very few test errors, so the metrics have wide uncertainty.
