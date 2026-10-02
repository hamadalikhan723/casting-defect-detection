import streamlit as st
from model import DefectClassifier

st.set_page_config(
    page_title="Casting Defect Detection",
    page_icon="🔍",
    layout="centered",
)

st.title("🔍 Casting Defect Detection")
st.write(
    "Upload a photo of a cast pump impeller and the AI model "
    "will classify it as normal or defective."
)

@st.cache_resource
def load_model():
    return DefectClassifier()

classifier = load_model()

uploaded_file = st.file_uploader(
    "Upload product image",
    type=["jpg", "jpeg", "png"],
)

threshold = st.slider(
    "Defect threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.05,
)

if uploaded_file is not None:
    from PIL import Image

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded product image",
        use_container_width=True,
    )

    if st.button("🔎 Analyze Image", type="primary"):
        with st.spinner("Analyzing image..."):
            result = classifier.predict(image, threshold)

        prediction = result["predicted_class"]
        confidence = result["confidence"]

        if prediction == "defective":
            st.error(f"⚠️ DEFECTIVE — {confidence * 100:.2f}% confidence")
        else:
            st.success(f"✅ NORMAL — {confidence * 100:.2f}% confidence")

        st.subheader("Prediction Details")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Prediction", prediction.upper())

        with col2:
            st.metric("Confidence", f"{confidence * 100:.2f}%")

        with col3:
            st.metric("Latency", f"{result['latency_ms']:.1f} ms")

        st.write(
            f"Probability of defect: "
            f"**{result['p_defective'] * 100:.2f}%**"
        )