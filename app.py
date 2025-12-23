import streamlit as st
import cv2
import numpy as np
from PIL import Image

from realtimedetection import detect_emotion_from_image

st.set_page_config(
    page_title="Emotion Detection System",
    layout="centered"
)

st.title("🎭 Emotion Detection System")
st.markdown("Detect human emotions using a trained CNN model")

# ================= UI OPTIONS =================
option = st.radio(
    "Choose Input Type:",
    ("Upload Image", "Live Webcam (Experimental)")
)

# ================= IMAGE UPLOAD =================
if option == "Upload Image":
    uploaded_file = st.file_uploader(
        "Upload a face image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img = np.array(image)

        results = detect_emotion_from_image(img)

        if len(results) == 0:
            st.warning("No face detected!")
        else:
            for (x, y, w, h, emotion) in results:
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
                cv2.putText(
                    img, emotion, (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2
                )

            st.image(img, channels="BGR", caption="Emotion Detected")

# ================= WEBCAM (CONTROLLED) =================
elif option == "Live Webcam (Experimental)":
    st.warning("Press START to activate webcam")

    start = st.button("▶ Start Webcam")
    stop = st.button("⏹ Stop Webcam")

    if start:
        cap = cv2.VideoCapture(0)
        frame_window = st.image([])

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = detect_emotion_from_image(frame)

            for (x, y, w, h, emotion) in results:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                cv2.putText(
                    frame, emotion, (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2
                )

            frame_window.image(frame, channels="BGR")

            if stop:
                break

        cap.release()
