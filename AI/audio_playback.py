import pandas as pd
import os
from playsound import playsound

def audio_playback(translated_file, audio_dir):
    """
    Menu-based system for playing audio files.
    Args:
        translated_file (str): Path to translated_summaries.csv.
        audio_dir (str): Directory containing audio files.
    """
    # Load translated summaries
    df = pd.read_csv(translated_file)
    
    # List available MovieIDs
    movie_ids = df['MovieID'].tolist()
    
    while True:
        print("\nAvailable MovieIDs:", movie_ids)
        print("Enter 'q' to quit.")
        movie_id = input("Enter MovieID: ")
        
        if movie_id.lower() == 'q':
            break
        
        if int(movie_id) not in movie_ids:
            print("Invalid MovieID. Try again.")
            continue
        
        # Display language options
        languages = ['Arabic', 'Urdu', 'Korean']
        print("\nAvailable languages:", languages)
        lang = input("Enter language: ").capitalize()
        
        if lang not in languages:
            print("Invalid language. Try again.")
            continue
        
        # Construct audio file path
        audio_file = os.path.join(audio_dir, str(movie_id), f"{lang.lower()}.mp3")
        
        if not os.path.exists(audio_file):
            print(f"Audio file not found for MovieID {movie_id}, {lang}.")
            continue
        
        try:
            print(f"Playing audio: {audio_file}")
            playsound(audio_file)
        except Exception as e:
            print(f"Error playing audio: {e}")

if __name__ == "__main__":
    translated_file = "data/translated_summaries.csv"
    audio_dir = "audio/summaries"
    
    audio_playback(translated_file, audio_dir)