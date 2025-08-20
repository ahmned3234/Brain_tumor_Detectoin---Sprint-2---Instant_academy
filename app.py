import streamlit as st
import numpy as np
import cv2
from PIL import Image
from skimage.transform import resize
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input # type:ignore

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    smooth = 1e-6
    intersection = tf.reduce_sum(y_true * y_pred)
    dice = (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)
    return bce + (1 - dice)

model = tf.keras.models.load_model(
    "brain_tumor_segementation.keras",
    custom_objects={"bce_dice_loss": bce_dice_loss}
)

model.save("brain_tumor_segementation_inference.keras", include_optimizer=False)

# --------------------------
# Load your models (cache so they load once)
# --------------------------
@st.cache_resource
def load_models():
    classification_model = tf.keras.models.load_model("Classification_Model.h5")
    segmentation_model = tf.keras.models.load_model("brain_tumor_segementation_inference.keras", compile=False)
    return classification_model, segmentation_model

classification_model, segmentation_model = load_models()

# --------------------------
# Helper functions
# --------------------------
# --- For Classification ---
def preprocess_image_classification(img, target_size):
    img = img.convert("RGB") 
    img = img.resize(target_size)
    img = np.array(img)
    return np.expand_dims(img, axis=0)

# --- For Segmentation ---
def preprocess_segmentation_image(image, target_size=(128, 128)):
    if isinstance(image, Image.Image):
        image = np.array(image)

    # resize to match training
    image = resize(image, target_size, mode='constant', preserve_range=True)

    # grayscale → RGB
    if len(image.shape) == 2:
        image = np.stack((image,) * 3, axis=-1)
    # RGBA → drop alpha
    elif image.shape[-1] == 4:
        image = image[..., :3]

    # normalize
    return image / 255.0

def predict_classification(img):
    img = preprocess_image_classification(img, (224,224))
    img = preprocess_input(img)
    pred = classification_model.predict(img)[0]
    label_idx = np.argmax(pred)                  
    confidence = float(pred[label_idx])           

    label = "Tumor Detected" if label_idx == 1 else "No Tumor"

    return label, confidence

def predict_segmentation(img):
    processed = preprocess_segmentation_image(img, (128,128))   
    processed_image = np.expand_dims(processed, axis=0)
    mask = segmentation_model.predict(processed_image)[0]
    pred_mask_binary = (mask > 0.5).astype(np.float32)

    return pred_mask_binary

# def overlay_mask(image_pil, mask, alpha=0.5):
#     """
#     Overlay tumor mask (red) on original MRI image.
#     """
#     image = np.array(image_pil.convert("RGB"))  # Ensure 3 channels
#     mask = mask.astype(np.uint8)

#     # Create red mask
#     red_mask = np.zeros_like(image)
#     red_mask[:, :, 0] = 255  # Red channel

#     # Apply overlay
#     overlay = np.where(mask[:, :, None] > 0,
#     cv2.addWeighted(image, 1 - alpha, red_mask, alpha, 0),
#     image)
#     return overlay

# --------------------------
# Streamlit App Layout
# --------------------------
st.set_page_config(page_title="Brain Tumor Detection", layout="wide")

st.title("🧠 Brain Tumor Detection App")
st.write("Upload an MRI scan to detect brain tumors via classification and segmentation.")

tab1, tab2, tab3 = st.tabs(["Classification", "Segmentation", "Both"])

# --- CLASSIFICATION TAB ---
with tab1:
    uploaded_file = st.file_uploader("Upload brain MRI image for classification", type=["jpg", "png", "jpeg"], key="clf")
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="🖼 Uploaded Image", width=300)

        if st.button("Predict", key="predict_clf"):
            label,conf = predict_classification(img)
            st.success(f"**Prediction:** {label} ({conf:.2f} confidence)")

# --- SEGMENTATION TAB ---
with tab2:
    uploaded_file = st.file_uploader("Upload brain MRI image for segmentation", type=["jpg", "png", "jpeg",".tif"], key="seg")
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="🖼 Uploaded Image", width=300)

        if st.button("Predict", key="predict_seg"):
            mask = predict_segmentation(img)
            st.image(mask, caption="🧩 Tumor Segmentation Result", width=300)

# --- BOTH TAB ---
with tab3:
    uploaded_file = st.file_uploader("Upload brain MRI image for both tasks", type=["jpg", "png", "jpeg"], key="both")
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="🖼 Uploaded Image", width=300)
        
        if st.button("Predict", key="predict_both"):
            # Classification
            label,conf = predict_classification(img)        
            
            # Segmentation
            mask = predict_segmentation(img)            
            # Results
            st.success(f"**Prediction:** {label} ({conf:.2f} confidence)")
            st.image(mask, caption="🧩 Segmentation Result", width=300)
