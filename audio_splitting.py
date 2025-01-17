import os
import librosa
import soundfile as sf

# --- Configurations ---
RAW_DATA_PATH = 'raw_data'  # Folder containing raw audio files
OUTPUT_PATH = 'training_data'  # Folder to save segmented audio files
SEGMENT_DURATION = 5.0  # Duration of each segment in seconds

# Ensure output directory exists
os.makedirs(OUTPUT_PATH, exist_ok=True)

def split_audio(file_path, output_path, segment_duration):
    """
    Splits an audio file into smaller segments and saves them.

    Args:
        file_path (str): Path to the input audio file.
        output_path (str): Path to the folder where segments will be saved.
        segment_duration (float): Duration of each segment in seconds.
    """
    # Load the audio file
    audio, sr = librosa.load(file_path, sr=None)
    
    # Calculate the number of samples per segment
    segment_samples = int(segment_duration * sr)

    # Split the audio into segments
    total_segments = len(audio) // segment_samples

    for i in range(total_segments):
        start_sample = i * segment_samples
        end_sample = start_sample + segment_samples

        segment = audio[start_sample:end_sample]

        # Save each segment
        output_file = os.path.join(output_path, f"{os.path.basename(file_path).split('.')[0]}_segment_{i + 1}.wav")
        sf.write(output_file, segment, sr)
        print(f"Saved: {output_file}")

    # Handle the last remaining segment (if any)
    remaining_samples = len(audio) % segment_samples
    if remaining_samples > 0:
        segment = audio[-remaining_samples:]
        output_file = os.path.join(output_path, f"{os.path.basename(file_path).split('.')[0]}_segment_last.wav")
        sf.write(output_file, segment, sr)
        print(f"Saved: {output_file}")

if __name__ == "__main__":
    # Process each file in the RAW_DATA_PATH directory
    audio_files = [f for f in os.listdir(RAW_DATA_PATH) if f.endswith('.wav')]

    for audio_file in audio_files:
        input_path = os.path.join(RAW_DATA_PATH, audio_file)
        split_audio(input_path, OUTPUT_PATH, SEGMENT_DURATION)

    print("All files processed and segments saved.")
