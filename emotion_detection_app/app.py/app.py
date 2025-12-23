import streamlit as st
import cv2
import numpy as np
from PIL import Image

from src.predict import predict_emotion
from src.face_detect import detect_faces

st.set_page_config(page_title="Emotion Detection")

st.title("😊 Emotion Detection Web App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img = np.array(image)

    faces = detect_faces(img)

    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        emotion = predict_emotion(face)

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(img, emotion, (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

    st.image(img, channels="BGR")
