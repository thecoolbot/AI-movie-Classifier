import os
import ast
import pickle
import pandas as pd
from extract_features import extract_tfidf_features
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

def prepare_labels(train_file, test_file, mlb_file):
    """
    Prepare multi-label genre labels.
    Args:
        train_file (str): Path to train_dataset.csv.
        test_file (str): Path to test_dataset.csv.
        mlb_file (str): Path to save the fitted MultiLabelBinarizer.
    Returns:
        tuple: (y_train, y_test, mlb) Binary labels and binarizer.
    """
    # Load datasets
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    
    # Convert string representations of lists to actual lists
    train_df['Genres_List'] = train_df['Genres_List'].apply(ast.literal_eval)
    test_df['Genres_List'] = test_df['Genres_List'].apply(ast.literal_eval)
    
    # Initialize MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    
    # Fit and transform on training genres
    y_train = mlb.fit_transform(train_df['Genres_List'])
    
    # Transform test genres
    y_test = mlb.transform(test_df['Genres_List'])
    
    # Save the binarizer
    with open(mlb_file, 'wb') as f:
        pickle.dump(mlb, f)
    
    return y_train, y_test, mlb

def train_model(X_train, y_train, model_file):
    """
    Train a multi-label Logistic Regression model.
    Args:
        X_train: TF-IDF features for training.
        y_train: Binary genre labels for training.
        model_file (str): Path to save the trained model.
    Returns:
        OneVsRestClassifier: Trained model.
    """
    # Initialize Logistic Regression with OneVsRestClassifier
    model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Save the model
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    
    return model

if __name__ == "__main__":
    train_file = "data/train_dataset.csv"
    test_file = "data/test_dataset.csv"
    mlb_file = "models/mlb.pkl"
    model_file = "models/logistic_regression_model.pkl"
    
    os.makedirs("models", exist_ok=True)
    
    # Load TF-IDF features (assumes extract_features.py has been run)
    X_train, X_test, vectorizer = extract_tfidf_features(train_file, test_file, "models/tfidf_vectorizer.pkl")
    
    # Prepare labels
    y_train, y_test, mlb = prepare_labels(train_file, test_file, mlb_file)
    print(f"Labels prepared: y_train shape = {y_train.shape}, y_test shape = {y_test.shape}")
    
    # Train model
    model = train_model(X_train, y_train, model_file)
    print(f"Model trained and saved to '{model_file}'.")