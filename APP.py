import streamlit as st 
import torch
from predict import load_image,load_model,predict
from streamlit_webrtc import webrtc_streamer
import av
import cv2
from PIL import Image
from dataset import test_transformer
transformer=test_transformer
model=load_model()
    
st.title("Eco Sort AI")


def video_processing_callback(frame):
    img = frame.to_ndarray(format="rgb24")

    pil_img = Image.fromarray(img)
    tensor_img=load_image(pil_img)
    prediction,confidence=predict(model=model,image=tensor_img)
    label = f"{prediction} ({confidence:.1f}%)"
    if confidence > 80:
     text_color = (0, 255, 0)    
    elif 50 <= confidence <= 80:
     text_color = (0, 255, 255)  
    else:
     text_color = (0, 0, 255)    
    cv2.putText(img, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

    return av.VideoFrame.from_ndarray(img, format="rgb24")

webrtc_streamer(
    key="vision-app",
    video_frame_callback=video_processing_callback
)

