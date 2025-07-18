import streamlit as st
import requests

# fastapi_url = "http://localhost:8000/" # THE FASTAPI SERVER IS LIVE
fastapi_url = "https://audiocompass-1ovq.onrender.com"

st.title("🎵 Audio Compass")

audio_file = st.file_uploader("Upload an Audio file", type=["mp3", "wav"], accept_multiple_files=False)

anal_check = st.button("🎯 Analyse") # LMAO

if anal_check:
    if not audio_file:
        st.error("🚨 Please upload an mp3 or wav audio file.")
    else:
        with st.spinner("⏳ Sending file to server and analysing…"):
            try:
                response = requests.post(
                    url=f"{fastapi_url}predict",
                    files={"file": audio_file}
                )
                if response.status_code == 200:
                    answer = response.json()
                    prediction = answer.get("prediction", {})
                    label = prediction.get("label", "Unknown")
                    tempo = prediction.get("tempo")
                    st.success(f"✅ Predicted Key & Scale: **{label}**")
                    st.success(f"✅ Predicted Tempo      : **{tempo}**")
                                     # tempo hehe

                else:
                    st.error(f"❌ Server returned status code {response.status_code}: {response.text}")

            except Exception as e:
                st.error(f"🔥 Error communicating with API: {e}")
   


