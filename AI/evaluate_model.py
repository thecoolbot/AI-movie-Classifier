import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MultiLabelBinarizer
import joblib
import warnings

# Ignore sklearn warnings about unknown labels during transformation
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

def load_data():
    X_train = joblib.load("X_train.pkl")
    y_train = joblib.load("y_train.pkl")
    X_test = joblib.load("X_test.pkl")
    y_test = joblib.load("y_test.pkl")
    return X_train, y_train, X_test, y_test

def load_model_and_mlb():
    model = joblib.load("model.pkl")
    mlb = joblib.load("mlb.pkl")
    return model, mlb

def print_metrics(name, y_true, y_pred):
    print(f"{name} Metrics:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision (micro):", precision_score(y_true, y_pred, average="micro", zero_division=0))
    print("Recall (micro):", recall_score(y_true, y_pred, average="micro", zero_division=0))
    print("F1-Score (micro):", f1_score(y_true, y_pred, average="micro", zero_division=0))
    print()

def evaluate_model(model, X_train, y_train, X_test, y_test, mlb, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)

    y_train_bin = mlb.transform(y_train)
    y_test_bin = mlb.transform(y_test)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print_metrics("Training", y_train_bin, y_train_pred)
    print_metrics("Test", y_test_bin, y_test_pred)

    # Compute confusion matrix for each genre
    for i, genre in enumerate(mlb.classes_):
        try:
            cm_train = confusion_matrix(y_train_bin[:, i], y_train_pred[:, i], labels=[0, 1])
            cm_test = confusion_matrix(y_test_bin[:, i], y_test_pred[:, i], labels=[0, 1])

            disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=[0, 1])
            disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=[0, 1])

            safe_genre = genre.replace("/", "_").replace("\\", "_").replace(" ", "_")  # sanitize filename
            plt.figure()
            disp_train.plot()
            plt.title(f"Train Confusion Matrix for {genre}")
            plt.savefig(os.path.join(output_dir, f"cm_train_{safe_genre}.png"))
            plt.close()

            plt.figure()
            disp_test.plot()
            plt.title(f"Test Confusion Matrix for {genre}")
            plt.savefig(os.path.join(output_dir, f"cm_test_{safe_genre}.png"))
            plt.close()

        except Exception as e:
            print(f"[Warning] Skipped confusion matrix for '{genre}': {e}")

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()
    model, mlb = load_model_and_mlb()
    evaluate_model(model, X_train, y_train, X_test, y_test, mlb)
