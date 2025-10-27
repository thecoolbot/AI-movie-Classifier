import pandas as pd
from gtts import gTTS
import os
import time

def convert_to_audio(input_file, audio_dir):
    """
    Convert translated summaries to audio files.
    Args:
        input_file (str): Path to translated_summaries.csv.
        audio_dir (str): Directory to save audio files.
    """
    # Load translated summaries
    df = pd.read_csv(input_file)
    
    # Create output directory
    os.makedirs(audio_dir, exist_ok=True)
    
    # Define languages
    languages = {'Arabic': 'ar', 'Urdu': 'ur', 'Korean': 'ko'}
    
    # Convert each summary to audio
    for idx, row in df.iterrows():
        movie_id = row['MovieID']
        # Create subdirectory for each MovieID
        movie_dir = os.path.join(audio_dir, str(movie_id))
        os.makedirs(movie_dir, exist_ok=True)
        
        for lang_name, lang_code in languages.items():
            text = row[f'Summary_{lang_name}']
            if not text or pd.isna(text):
                print(f"Skipping empty translation for MovieID {movie_id}, {lang_name}")
                continue
            try:
                tts = gTTS(text=text, lang=lang_code)
                output_file = os.path.join(movie_dir, f"{lang_name.lower()}.mp3")
                tts.save(output_file)
                print(f"Generated audio: {output_file}")
                time.sleep(1)  # Avoid API rate limits
            except Exception as e:
                print(f"Error generating audio for MovieID {movie_id}, {lang_name}: {e}")

if __name__ == "__main__":
    input_file = "data/translated_summaries.csv"
    audio_dir = "audio/summaries"
    
    convert_to_audio(input_file, audio_dir)
    print(f"Audio files saved in '{audio_dir}'.")