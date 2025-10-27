import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_summary(text):
    """
    Clean and preprocess a movie summary.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    cleaned_tokens = [
        lemmatizer.lemmatize(token) for token in tokens
        if token not in stop_words and len(token) > 2
    ]
    return ' '.join(cleaned_tokens)

def process_data(input_file, output_dir):
    """
    Process movie data and split into train/test sets.
    """
    try:
        df = pd.read_csv(input_file)
        if 'summary' not in df.columns or 'genres' not in df.columns:
            raise ValueError("Input CSV must have 'summary' and 'genres' columns")
        
        df['cleaned_summary'] = df['summary'].apply(clean_summary)
        df['genres'] = df['genres'].apply(lambda x: eval(x) if isinstance(x, str) else x)
        
        # Split into train (80%) and test (20%)
        train_df = df.sample(frac=0.8, random_state=42)
        test_df = df.drop(train_df.index)
        
        # Save datasets
        os.makedirs(output_dir, exist_ok=True)
        train_df.to_csv(os.path.join(output_dir, 'train_dataset.csv'), index=False)
        test_df.to_csv(os.path.join(output_dir, 'test_dataset.csv'), index=False)
        print("Train and test datasets saved successfully.")
    except Exception as e:
        print(f"Error processing data: {str(e)}")

if __name__ == "__main__":
    input_file = "data/raw_movie_data.csv"  # Adjust to your input file
    output_dir = "data"
    process_data(input_file, output_dir)