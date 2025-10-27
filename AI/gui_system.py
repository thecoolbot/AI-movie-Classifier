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
import tkinter as tk
from tkinter import messagebox, scrolledtext

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
        return f"Error translating to {lang_code}: {e}"

def generate_and_play_audio(summary, lang_code, lang_name):
    """
    Generate and play audio for a summary using pygame.
    Args:
        summary (str): Text to convert to audio.
        lang_code (str): Language code.
        lang_name (str): Language name for file naming.
    Returns:
        str: Success or error message.
    """
    if not summary:
        return "No text to convert to audio."
    
    os.makedirs("temp", exist_ok=True)
    audio_file = f"temp/{lang_name.lower()}_summary.mp3"
    
    try:
        tts = gTTS(text=summary, lang=lang_code)
        tts.save(audio_file)
        pygame.mixer.init()
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        pygame.mixer.quit()
        time.sleep(1)  # Avoid API rate limits
        return f"Playing audio: {audio_file}"
    except Exception as e:
        return f"Error generating or playing audio: {e}"

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
        return ["No genres predicted"]
    
    X = vectorizer.transform([summary])
    y_pred = model.predict(X)
    genres = mlb.inverse_transform(y_pred)[0]
    return list(genres) if genres else ["No genres predicted"]

class FilmceptionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Filmception: Movie Summary Translator and Genre Classifier")
        self.root.geometry("600x500")
        
        # Load models
        try:
            with open("models/tfidf_vectorizer.pkl", 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open("models/mlb.pkl", 'rb') as f:
                self.mlb = pickle.load(f)
            with open("models/logistic_regression_model.pkl", 'rb') as f:
                self.model = pickle.load(f)
        except FileNotFoundError as e:
            messagebox.showerror("Error", f"Model or vectorizer file not found: {e}")
            self.root.destroy()
            return
        
        # Language options
        self.languages = {'Arabic': 'ar', 'Urdu': 'ur', 'Korean': 'ko'}
        
        # GUI elements
        tk.Label(root, text="Enter Movie Summary:", font=("Arial", 12)).pack(pady=10)
        
        self.summary_text = scrolledtext.ScrolledText(root, height=5, width=50, wrap=tk.WORD)
        self.summary_text.pack(pady=10)
        
        tk.Label(root, text="Select Action:", font=("Arial", 12)).pack(pady=10)
        
        tk.Button(root, text="Translate and Play Audio (Arabic)", command=lambda: self.process_audio('Arabic')).pack(pady=5)
        tk.Button(root, text="Translate and Play Audio (Urdu)", command=lambda: self.process_audio('Urdu')).pack(pady=5)
        tk.Button(root, text="Translate and Play Audio (Korean)", command=lambda: self.process_audio('Korean')).pack(pady=5)
        tk.Button(root, text="Predict Genres", command=self.predict).pack(pady=5)
        tk.Button(root, text="Exit", command=self.exit).pack(pady=5)
        
        tk.Label(root, text="Output:", font=("Arial", 12)).pack(pady=10)
        self.output_text = scrolledtext.ScrolledText(root, height=8, width=50, wrap=tk.WORD, state='disabled')
        self.output_text.pack(pady=10)
    
    def process_audio(self, lang_name):
        """
        Process summary for translation and audio playback.
        """
        summary = self.summary_text.get("1.0", tk.END).strip()
        if not summary:
            messagebox.showerror("Error", "Summary cannot be empty.")
            return
        
        self.output_text.configure(state='normal')
        self.output_text.delete("1.0", tk.END)
        
        if lang_name not in self.languages:
            self.output_text.insert(tk.END, f"Invalid language: {lang_name}\n")
            self.output_text.configure(state='disabled')
            return
        
        lang_code = self.languages[lang_name]
        translated_summary = translate_summary(summary, lang_code)
        if translated_summary.startswith("Error"):
            self.output_text.insert(tk.END, translated_summary + "\n")
        else:
            self.output_text.insert(tk.END, f"Translated summary ({lang_name}): {translated_summary}\n")
            audio_result = generate_and_play_audio(translated_summary, lang_code, lang_name)
            self.output_text.insert(tk.END, audio_result + "\n")
        
        self.output_text.configure(state='disabled')
    
    def predict(self):
        """
        Predict genres for the input summary.
        """
        summary = self.summary_text.get("1.0", tk.END).strip()
        if not summary:
            messagebox.showerror("Error", "Summary cannot be empty.")
            return
        
        cleaned_summary = clean_summary(summary)
        genres = predict_genres(cleaned_summary, self.vectorizer, self.model, self.mlb)
        
        self.output_text.configure(state='normal')
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, "Predicted Genres: " + ", ".join(genres) + "\n")
        self.output_text.configure(state='disabled')
    
    def exit(self):
        """
        Exit the application.
        """
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FilmceptionGUI(root)
    root.mainloop()