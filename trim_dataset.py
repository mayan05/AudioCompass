import os
import traceback
from tqdm import tqdm
from pydub import AudioSegment

def trim_dataset(input_folder, output_folder, chunk_duration=15, overlap=0):
    os.makedirs(output_folder, exist_ok=True)
    audio_extensions = {'.wav', '.mp3', '.webm'}
    audio_files = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if any(file.endswith(ext) for ext in audio_extensions):
                audio_files.append(os.path.join(root, file))
    print(f"Found {len(audio_files)} audio files to process")
    total_chunks = 0
    for audio_file_path in tqdm(audio_files, desc="Processing audio files"):
        try:
            # Load audio using pydub
            audio = AudioSegment.from_file(audio_file_path)
            sr = audio.frame_rate # Get sample rate from pydub audio object
            y = audio.get_array_of_samples() # Get raw samples as a Python array

            chunk_milliseconds = int(chunk_duration * 1000)
            overlap_milliseconds = int(overlap * 1000)
            step_milliseconds = chunk_milliseconds - overlap_milliseconds
            
            total_duration = len(audio) / 1000 # Duration in seconds
            base_name = os.path.splitext(os.path.basename(audio_file_path))[0]
            chunk_count = 0
            
            # Iterate through audio and create chunks
            for i in range(0, len(audio) - chunk_milliseconds + 1, step_milliseconds):
                chunk = audio[i:i + chunk_milliseconds]
                chunk_count += 1
                output_filename = f"{base_name}_{chunk_count}.mp3"
                output_path = os.path.join(output_folder, output_filename)
                
                # Export chunk using pydub
                chunk.export(output_path, format="mp3", bitrate="192k")
                total_chunks += 1
            print(f"✓ {os.path.basename(audio_file_path)}: {total_duration:.1f}s → {chunk_count} chunks")
        except Exception as e:
            print(f"✗ Error processing {os.path.basename(audio_file_path)}: {traceback.format_exc()}")
    print(f"\nProcessing complete!")
    print(f"Total chunks created: {total_chunks}")

input_folder = "./remaining_tracks"
output_folder = "./trimmed_audio_mp3"

trim_dataset(input_folder, output_folder, chunk_duration=15)