import pandas as pd
import re
import spacy
from tqdm import tqdm

# Load spaCy English model (we use it for stopword removal, tokenization, lemmatization)
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def preprocess_summary(text):
    """
    Clean and preprocess a movie summary according to specified requirements:
    - Lowercase text
    - Remove numbers and special characters
    - Tokenization
    - Stopword removal
    - Lemmatization
    - Remove short/redundant tokens
    """
    # 1. Lowercasing and basic cleanup
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)          # Remove special chars, numbers
    text = re.sub(r'\s+', ' ', text).strip()       # Remove extra spaces

    # 2. Tokenization + Stopword removal + Lemmatization
    doc = nlp(text)
    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop and token.is_alpha and len(token) > 2
    ]

    return ' '.join(tokens)

def load_and_process_summaries(file_path):
    """
    Load the plot summaries file and apply preprocessing.
    """
    # Load tab-separated data with two columns: MovieID and Summary
    df = pd.read_csv(file_path, sep='\t', names=['MovieID', 'Summary'])

    # Apply preprocessing with progress bar
    tqdm.pandas(desc="🔄 Cleaning Summaries")
    df['Cleaned_Summary'] = df['Summary'].progress_apply(preprocess_summary)

    return df

# Run the script
if __name__ == "__main__":
    file_path = "data/plot_summaries.txt"  # Change path if needed
    cleaned_df = load_and_process_summaries(file_path)

    # Save to CSV
    cleaned_df.to_csv("data/cleaned_summaries.csv", index=False)
    print("✅ Preprocessing complete. Cleaned file saved as 'cleaned_summaries.csv'")
