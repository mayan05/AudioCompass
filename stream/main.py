import streamlit as st
import requests

fastapi_url = "http://localhost:8000/"

st.title("Audio Compass")

audio_file = st.file_uploader("Upload an Audio file", type=["mp3", "wav"], accept_multiple_files=False)

anal_check = st.button("Analyse") # LMAO
try:
    if not audio_file and anal_check:
        st.error("Please upload an wav/mp3 audio file", icon="🚨")
    elif audio_file and anal_check: 
        try:
            response = requests.get(fastapi_url)
            answer = response.json()
            st.write(answer['message'])
        except Exception as e:
            st.error(f"Error from API: {e}")
        
except Exception as e:
    st.error(f"Error: {e}")
   


