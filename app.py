import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import pandas as pd
from io import BytesIO
import cv2


st.set_page_config(
    page_title="VISION - Facial Expression Recognition",
    page_icon="😊"
)


MODEL_PATH = "vision_emotion_model.keras"
CLASSES_PATH = "vision_emotion_classes.json"
IMG_SIZE = (48, 48)
UNKNOWN_THRESHOLD = 0.60


@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASSES_PATH, "r") as f:
        class_names = json.load(f)

    return model, class_names


def crop_face(image):
    """
    Detects the largest face in the image and returns a cropped grayscale face.
    If no face is found, returns None.
    """
    image_rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda face: face[2] * face[3])

    padding = int(0.15 * w)

    x1 = max(x - padding, 0)
    y1 = max(y - padding, 0)
    x2 = min(x + w + padding, gray.shape[1])
    y2 = min(y + h + padding, gray.shape[0])

    face = gray[y1:y2, x1:x2]
    face_image = Image.fromarray(face)

    return face_image


def preprocess_image(image):
    """
    Crops the face, resizes it to 48x48, normalizes it,
    and prepares it for the CNN model.
    """
    face_image = crop_face(image)

    if face_image is None:
        return None, None

    face_image = face_image.resize(IMG_SIZE)

    img_array = np.array(face_image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array, face_image


def image_to_bytes(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


model, class_names = load_model_and_classes()

st.title("VISION – Facial Expression Recognition")
st.write("Upload or take a face image, and the system will classify the facial expression.")

source = st.radio(
    "Choose image source:",
    ["Camera", "Upload image"]
)

uploaded_image = None

if source == "Camera":
    uploaded_image = st.camera_input("Take a picture")
else:
    uploaded_image = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded image", use_container_width=True)

    st.download_button(
        label="Save image",
        data=image_to_bytes(image),
        file_name="uploaded_image.png",
        mime="image/png"
    )

    if st.button("Classify"):
        processed_image, face_image = preprocess_image(image)

        if processed_image is None:
            st.error("No face detected. Try another image with a clear front-facing face.")
        else:
            st.image(face_image, caption="Detected face used by the model", width=200)

            predictions = model.predict(processed_image)[0]

            best_index = np.argmax(predictions)
            confidence = predictions[best_index]
            predicted_class = class_names[best_index]

            if confidence < UNKNOWN_THRESHOLD:
                st.warning(f"Unknown — confidence is too low: {confidence:.2%}")
            else:
                st.success(f"Prediction: {predicted_class}")
                st.write(f"Confidence: {confidence:.2%}")

            results_df = pd.DataFrame({
                "Emotion": class_names,
                "Probability": predictions * 100
            }).sort_values(by="Probability", ascending=False)

            st.subheader("Probabilities")
            st.dataframe(results_df, use_container_width=True)
            st.bar_chart(results_df.set_index("Emotion"))