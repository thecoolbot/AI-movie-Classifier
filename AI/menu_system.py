import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from googletrans import Translator
from gtts import gTTS
import pygame
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
import os
import time

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_summary(text):
    """
    Clean and preprocess a movie summary using NLTK.
    Args:
        text (str): Raw movie summary.
    Returns:
        str: Cleaned and preprocessed summary.
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    cleaned_tokens = [
        lemmatizer.lemmatize(token) for token in tokens
        if token not in stop_words and len(token) > 2
    ]
    return ' '.join(cleaned_tokens)

def translate_summary(summary, lang_code):
    """
    Translate a summary into the specified language.
    Args:
        summary (str): Input summary.
        lang_code (str): Target language code (e.g., 'ar', 'ur', 'ko').
    Returns:
        str: Translated summary.
    """
    translator = Translator()
    try:
        translated = translator.translate(summary, dest=lang_code)
        return translated.text
    except Exception as e:
        print(f"Error translating to {lang_code}: {e}")
        return ""

def generate_and_play_audio(summary, lang_code, lang_name):
    """
    Generate and play audio for a summary using pygame.
    Args:
        summary (str): Text to convert to audio.
        lang_code (str): Language code.
        lang_name (str): Language name for file naming.
    """
    if not summary:
        print("No text to convert to audio.")
        return
    
    os.makedirs("temp", exist_ok=True)
    audio_file = f"temp/{lang_name.lower()}_summary.mp3"
    
    try:
        tts = gTTS(text=summary, lang=lang_code)
        tts.save(audio_file)
        pygame.mixer.init()
        pygame.mixer.music.load(audio_file)
        print(f"Playing audio: {audio_file}")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        pygame.mixer.quit()
        time.sleep(1)  # Avoid API rate limits
    except Exception as e:
        print(f"Error generating or playing audio: {e}")

def predict_genres(summary, vectorizer, model, mlb):
    """
    Predict genres for a given summary.
    Args:
        summary (str): Cleaned movie summary.
        vectorizer: Fitted TfidfVectorizer.
        model: Trained OneVsRestClassifier.
        mlb: Fitted MultiLabelBinarizer.
    Returns:
        list: Predicted genres.
    """
    if not summary:
        return []
    
    X = vectorizer.transform([summary])
    y_pred = model.predict(X)
    genres = mlb.inverse_transform(y_pred)[0]
    return list(genres) if genres else ["No genres predicted"]

def main():
    """
    Main function for the menu-based system.
    """
    try:
        with open("models/tfidf_vectorizer.pkl", 'rb') as f:
            vectorizer = pickle.load(f)
        with open("models/mlb.pkl", 'rb') as f:
            mlb = pickle.load(f)
        with open("models/logistic_regression_model.pkl", 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error: Model or vectorizer file not found: {e}")
        return
    
    print("Welcome to Filmception: AI-Powered Movie Summary Translator and Genre Classifier")
    
    # Get user input for summary
    while True:
        print("\nPrompting for summary input...")
        summary = input("Enter a movie summary (or 'q' to quit): ").strip()
        print(f"Received input: '{summary}'")
        if summary.lower() == 'q':
            print("Exiting Filmception. Goodbye!")
            return
        if not summary:
            print("Error: Summary cannot be empty. Please try again.")
            continue
        break
    
    cleaned_summary = clean_summary(summary)
    
    # Menu loop
    languages = {'Arabic': 'ar', 'Urdu': 'ur', 'Korean': 'ko'}
    while True:
        print("\nMenu:")
        print("1. Convert Summary to Audio")
        print("2. Predict Genre")
        print("3. Exit")
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == '1':
            print("\nAvailable languages:", list(languages.keys()))
            lang_name = input("Enter language: ").capitalize()
            if lang_name not in languages:
                print("Error: Invalid language. Please choose from", list(languages.keys()))
                continue
            
            lang_code = languages[lang_name]
            translated_summary = translate_summary(summary, lang_code)
            if translated_summary:
                print(f"Translated summary ({lang_name}): {translated_summary}")
                generate_and_play_audio(translated_summary, lang_code, lang_name)
            else:
                print("Translation failed. Please try again.")
        
        elif choice == '2':
            genres = predict_genres(cleaned_summary, vectorizer, model, mlb)
            print("\nPredicted Genres:", ", ".join(genres) if genres else "None")
        
        elif choice == '3':
            print("Exiting Filmception. Goodbye!")
            break
        
        else:
            print("Error: Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()