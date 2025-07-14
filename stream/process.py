import librosa
import numpy as np
from pydub import AudioSegment
import os
from fastapi import UploadFile
import tempfile
import shutil

def trim_and_extract_features(file: UploadFile, chunk_duration=15, sample_rate=22050, n_mels=150):
    temp_audio_file_path = None
    mp3_path = None
    try:
        # Save the uploaded file to a temporary location
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            temp_audio_file_path = tmp_file.name

        # Check if the file is already an MP3
        if suffix.lower() == ".mp3":
            mp3_path = temp_audio_file_path
        else:
            # Convert to mp3 if not already mp3
            mp3_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
            audio = AudioSegment.from_file(temp_audio_file_path)
            audio.export(mp3_path, format="mp3")
        
        y, sr = librosa.load(mp3_path, sr=sample_rate)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        chunk_length = int(chunk_duration * sr)
        total_samples = len(y)
        input_features = []
        
        for start in range(0, total_samples, chunk_length):
            end = min(start + chunk_length, total_samples)
            chunk = y[start:end]
            if len(chunk) < chunk_length:
                # Pad the last chunk if it's shorter than chunk_length
                chunk = np.pad(chunk, (0, chunk_length - len(chunk)))
            S = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=n_mels)
            log_S = librosa.power_to_db(S, ref=np.max)
            input_features.append(log_S)
        return (np.array(input_features), tempo)
    
    finally:
        # Clean up temporary files
        if temp_audio_file_path and os.path.exists(temp_audio_file_path):
            os.remove(temp_audio_file_path)
        # Only delete mp3_path if it's a newly created file, not if it's the original temp_audio_file_path
        if mp3_path and os.path.exists(mp3_path) and mp3_path != temp_audio_file_path:
            os.remove(mp3_path)