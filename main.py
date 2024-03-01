import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

from deepface import DeepFace
from PIL import Image

def crop_image(image, face_region):
    x = face_region['x']
    y = face_region['y']
    w = face_region['w']
    h = face_region['h']

    return image.crop((x, y, x+w, y+h))

if 'uploaded_image' not in st.session_state:
    st.session_state['uploaded_image'] = None
 
if 'detected_faces' not in st.session_state:
    st.session_state['detected_faces'] = None
    
with st.sidebar:
    uploaded_image = st.file_uploader(label="Please Upload an Image", type=['PNG', 'JPEG'])
    if uploaded_image is not None:
        st.image(uploaded_image)

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert('RGB')
    
    try:
        if st.session_state['uploaded_image'] != uploaded_image.file_id:
            st.session_state['detected_faces'] = DeepFace.analyze(img_path = np.asarray(image))
            st.session_state['uploaded_image'] = uploaded_image.file_id
        
        face_options = [i for i in range(0, len(st.session_state['detected_faces']))]
        face_option = st.selectbox("Select a Face to Analyze", face_options)
        face_data = st.session_state['detected_faces'][face_option]
        
        st.image(crop_image(image, face_region=face_data['region']))    

        tab1, tab2, tab3, tab4 = st.tabs(["Age and Gender", "Race", "Emotion", "Full Report"])
        
        with tab1:
            st.write("Age: ", face_data['age'])
            st.write("Gender: ", face_data['dominant_gender'])
            st.write(face_data['gender'])
            
        with tab2:
            st.write("Race: ", face_data['dominant_race'])
            st.write(face_data['race'])
            
        with tab3:
            st.write("Emotion: ", face_data['dominant_emotion'])
            st.write(face_data['emotion'])
        
        with tab4:
            st.write(face_data)
            
    except:
        st.write("Face could not be detected. Please try a different image.")