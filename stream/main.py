import streamlit as st

st.title("Audio Compass")

audio_file = st.file_uploader("Upload an Audio file", type=["wav", "mp3"], accept_multiple_files=False)

anal_click = st.button("Analyse") # LMOA
try:
    if not audio_file and anal_click:
        st.error("Please upload an wav/mp3 audio file", icon="🚨")
    elif audio_file and anal_click:
        st.write("Analyzing...")

except Exception as e:
    st.error(f"Error: {e}")
   


