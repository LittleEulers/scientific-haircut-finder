import streamlit as st
import mediapipe as mp
import cv2
import numpy as np

st.set_page_config(page_title="2026 Haircut Scientist", layout="wide")
st.title("🔬 Scientific Haircut Analysis")
st.write("Using 468 facial landmarks to calculate your 2026 trend match.")

uploaded_file = st.file_uploader("Upload a clear front-facing photo", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    # Convert file to opencv image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, channels="BGR", caption="Input Data", width=400)

    # MediaPipe Logic
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            # Calculate Ratio (Height/Width)
            face_height = landmarks[152].y - landmarks[10].y
            face_width = landmarks[454].x - landmarks[234].x
            ratio = face_height / face_width
            
            st.metric("Facial Geometric Ratio", f"{ratio:.2f}")
            
            # 2026 Trend Logic
            if ratio < 1.25:
                st.header("Recommendation: The 'Power Bob'")
                st.write("Your rounder facial geometry benefits from the sharp, horizontal lines of the 2026 blunt bob trend.")
            else:
                st.header("Recommendation: 'Birkin Bangs'")
                st.write("Your elongated facial structure is perfectly balanced by the 2026 wispy fringe trend.")
        else:
            st.error("Science failed: No face detected. Try better lighting!")
