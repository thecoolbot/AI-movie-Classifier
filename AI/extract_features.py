import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

def extract_tfidf_features(train_file, test_file, vectorizer_file):
    """
    Extract TF-IDF features from movie summaries.
    Args:
        train_file (str): Path to train_dataset.csv.
        test_file (str): Path to test_dataset.csv.
        vectorizer_file (str): Path to save the fitted TfidfVectorizer.
    Returns:
        tuple: (X_train, X_test, vectorizer) TF-IDF features and vectorizer.
    """
    # Load datasets
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    
    # Extract summaries
    train_summaries = train_df['Cleaned_Summary'].fillna('')
    test_summaries = test_df['Cleaned_Summary'].fillna('')
    
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    
    # Fit and transform on training data
    X_train = vectorizer.fit_transform(train_summaries)
    
    # Transform test data (using the same vectorizer)
    X_test = vectorizer.transform(test_summaries)
    
    # Save the vectorizer for later use
    with open(vectorizer_file, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    return X_train, X_test, vectorizer

if __name__ == "__main__":
    train_file = "data/train_dataset.csv"
    test_file = "data/test_dataset.csv"
    vectorizer_file = "models/tfidf_vectorizer.pkl"
    
    os.makedirs("models", exist_ok=True)
    X_train, X_test, vectorizer = extract_tfidf_features(train_file, test_file, vectorizer_file)
    print(f"TF-IDF features extracted: X_train shape = {X_train.shape}, X_test shape = {X_test.shape}")
    print(f"Vectorizer saved to '{vectorizer_file}'.")