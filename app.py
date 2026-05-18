# =========================
# Imports
# =========================

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import pandas as pd
from io import BytesIO
import cv2


# =========================
# Streamlit page settings
# Must be near the top of the file
# =========================

st.set_page_config(
    page_title="VISION - Facial Expression Recognition",
    page_icon="😊"
)


# =========================
# Constants / Settings
# =========================

# The trained CNN model file
MODEL_PATH = "vision_emotion_model.keras"

# The JSON file that contains the class names
CLASSES_PATH = "vision_emotion_classes.json"

# The image size that the model expects
# FER2013 images are 48x48
IMG_SIZE = (48, 48)

# If confidence is lower than 60%, we show "Unknown"
UNKNOWN_THRESHOLD = 0.60


# =========================
# Load model and class names
# =========================

@st.cache_resource
def load_model_and_classes():
    """
    Loads the trained model and the class names.

    st.cache_resource makes Streamlit load the model only once,
    instead of loading it again every time the page refreshes.
    """

    # Load the saved Keras model
    model = tf.keras.models.load_model(MODEL_PATH)

    # Load class names from the JSON file
    with open(CLASSES_PATH, "r") as f:
        class_names = json.load(f)

    return model, class_names


# =========================
# Face detection function
# =========================

def crop_face(image):
    """
    Detects the largest face in the uploaded image.

    The model was trained on cropped face images,
    so we should crop the face before prediction.

    If no face is detected, the function returns None.
    """

    # Convert PIL image to RGB NumPy array
    image_rgb = np.array(image.convert("RGB"))

    # Convert RGB image to grayscale
    # Face detection works better on grayscale images
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # Load OpenCV's built-in face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # If no faces were detected, return None
    if len(faces) == 0:
        return None

    # Choose the largest detected face
    # This helps if there is more than one face in the image
    x, y, w, h = max(faces, key=lambda face: face[2] * face[3])

    # Add padding around the face so we do not crop too tightly
    padding = int(0.15 * w)

    # Make sure the crop does not go outside the image borders
    x1 = max(x - padding, 0)
    y1 = max(y - padding, 0)
    x2 = min(x + w + padding, gray.shape[1])
    y2 = min(y + h + padding, gray.shape[0])

    # Crop the face from the grayscale image
    face = gray[y1:y2, x1:x2]

    # Convert the cropped NumPy array back to a PIL image
    face_image = Image.fromarray(face)

    return face_image


# =========================
# Image preprocessing
# =========================

def preprocess_image(image):
    """
    Prepares the image before sending it to the model.

    Steps:
    1. Detect and crop the face
    2. Resize to 48x48
    3. Normalize pixel values to range 0-1
    4. Add dimensions so the shape matches the model input

    Final shape:
    (1, 48, 48, 1)
    """

    # Crop the face first
    face_image = crop_face(image)

    # If no face was found, return None
    if face_image is None:
        return None, None

    # Resize the face to the size expected by the model
    face_image = face_image.resize(IMG_SIZE)

    # Convert image to NumPy array and normalize values
    img_array = np.array(face_image).astype("float32") / 255.0

    # Add channel dimension: (48, 48) -> (48, 48, 1)
    img_array = np.expand_dims(img_array, axis=-1)

    # Add batch dimension: (48, 48, 1) -> (1, 48, 48, 1)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array, face_image


# =========================
# Convert image to bytes
# Used for the "Save image" button
# =========================

def image_to_bytes(image):
    """
    Converts a PIL image into bytes,
    so Streamlit can download/save it.
    """

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


# =========================
# Load the model
# =========================

model, class_names = load_model_and_classes()


# =========================
# Streamlit UI
# =========================

st.title("VISION – Facial Expression Recognition")

st.write(
    "Upload or take a face image, and the system will classify the facial expression."
)

# User chooses image source
source = st.radio(
    "Choose image source:",
    ["Camera", "Upload image"]
)

uploaded_image = None

# Camera input
if source == "Camera":
    uploaded_image = st.camera_input("Take a picture")

# File upload input
else:
    uploaded_image = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )


# =========================
# If the user uploaded/took an image
# =========================

if uploaded_image is not None:

    # Open the uploaded image using PIL
    image = Image.open(uploaded_image)

    # Show the original image
    st.image(
        image,
        caption="Uploaded image",
        use_container_width=True
    )

    # Button for saving/downloading the uploaded image
    st.download_button(
        label="Save image",
        data=image_to_bytes(image),
        file_name="uploaded_image.png",
        mime="image/png"
    )

    # Classification button
    if st.button("Classify"):

        # Preprocess image before prediction
        processed_image, face_image = preprocess_image(image)

        # If no face was detected
        if processed_image is None:
            st.error(
                "No face detected. Try another image with a clear front-facing face."
            )

        else:
            # Show the cropped face that the model actually receives
            st.image(
                face_image,
                caption="Detected face used by the model",
                width=200
            )

            # Predict emotion
            predictions = model.predict(processed_image)[0]

            # Get the index of the highest probability
            best_index = np.argmax(predictions)

            # Get confidence value
            confidence = predictions[best_index]

            # Get predicted class name
            predicted_class = class_names[best_index]

            # If confidence is too low, show Unknown
            if confidence < UNKNOWN_THRESHOLD:
                st.warning(
                    f"Unknown — confidence is too low: {confidence:.2%}"
                )

            # Otherwise, show the predicted emotion
            else:
                st.success(f"Prediction: {predicted_class}")
                st.write(f"Confidence: {confidence:.2%}")

            # Create a table with all class probabilities
            results_df = pd.DataFrame({
                "Emotion": class_names,
                "Probability": predictions * 100
            }).sort_values(
                by="Probability",
                ascending=False
            )

            # Show probability table
            st.subheader("Probabilities")
            st.dataframe(results_df, use_container_width=True)

            # Show probability chart
            st.bar_chart(results_df.set_index("Emotion"))