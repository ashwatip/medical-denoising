import streamlit as st
import torch
import numpy as np
from PIL import Image
from model import DenoisingAutoencoder
from scipy.ndimage import uniform_filter

st.set_page_config(page_title="Medical Image Denoiser", layout="wide")

# load the trained model
model = DenoisingAutoencoder()
model.load_state_dict(torch.load("model_weights.pt"))
model.eval()

st.title("Medical Image Denoiser")
st.write("Upload a CT scan image, pick a corruption type, and the neural network will reconstruct it.")

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
corruption = st.selectbox("Corruption type", ["Noise", "Blur", "Missing Pixels"])

if uploaded is not None:
    img = Image.open(uploaded).convert("L").resize((64, 64))
    original = np.asarray(img, dtype=np.float32) / 255.0

    # retry button just reruns the app with a new random seed
    if st.button("Retry"):
        st.rerun()

    if corruption == "Noise":
        noise = np.random.uniform(-0.1, 0.1, size=original.shape)
        corrupted = np.clip(original + noise, 0, 1)
    elif corruption == "Blur":
        corrupted = uniform_filter(original, size=5)
    else:
        mask = (np.random.rand(*original.shape) > 0.3)
        corrupted = original * mask

    input_tensor = torch.tensor(corrupted, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
    reconstructed = output.squeeze().numpy()

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Corrupted")
        st.image(corrupted, clamp=True, use_container_width=True)
        st.caption(f"What the model receives, simulated {corruption.lower()}")

    with col2:
        st.subheader("Reconstructed")
        st.image(reconstructed, clamp=True, use_container_width=True)
        st.caption("Cleaned up version")

    with col3:
        st.subheader("Original")
        st.image(original, clamp=True, use_container_width=True)
        st.caption("What we're trying to recover")

    st.markdown("---")
    st.markdown("**How it works:** The model was trained on 34,000 real CT scan slices from the OrganAMNIST dataset. It learned to map corrupted images back to clean originals by practicing on thousands of corrupted/clean pairs.")