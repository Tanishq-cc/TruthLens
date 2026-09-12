"""
TruthLens - Fake News Classifier
GDG JIIT 128 AI/ML Core Team task

Dataset: WELFake_Dataset.csv
Expected columns:
    title, text, label
WELFake label convention:
    0 = Fake
    1 = Real
"""

import os
import re
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

DATA_PATH = "data/WELFake_Dataset.csv"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("assets", exist_ok=True)


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            "data/WELFake_Dataset.csv not found. Download the WELFake dataset "
            "and put the CSV inside the data folder."
        )

    df = pd.read_csv(DATA_PATH)

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("The CSV must contain 'text' and 'label' columns.")

    df = df.dropna(subset=["text", "label"]).copy()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # WELFake: 0 = Fake, 1 = Real.
    if "title" in df.columns:
        df["combined_text"] = (
            df["title"].fillna("").astype(str) + " " +
            df["text"].fillna("").astype(str)
        )
    else:
        df["combined_text"] = df["text"].astype(str)

    df["clean_text"] = df["combined_text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0]

    return df


def evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC()
    }

    results = []
    trained = {}

    for name, classifier in models.items():
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                stop_words="english",
                max_features=50000,
                ngram_range=(1, 2)
            )),
            ("classifier", classifier)
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions, zero_division=0),
            "Recall": recall_score(y_test, predictions, zero_division=0),
            "F1 Score": f1_score(y_test, predictions, zero_division=0)
        })
        trained[name] = (pipeline, predictions)

    results_df = pd.DataFrame(results).sort_values("F1 Score", ascending=False)
    return results_df, trained


def main():
    print("Loading WELFake dataset...")
    df = load_data()

    print("\nDataset shape:", df.shape)
    print("\nColumns:", list(df.columns))
    print("\nLabel distribution:")
    print(df["label"].value_counts().sort_index())
    print("\nLabel meaning: 0 = Fake, 1 = Real")

    # EDA
    label_counts = df["label"].value_counts().sort_index()
    labels = ["Fake (0)", "Real (1)"]
    values = [label_counts.get(0, 0), label_counts.get(1, 0)]

    plt.figure(figsize=(7, 4))
    plt.bar(labels, values)
    plt.title("Fake vs Real News Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Articles")
    plt.tight_layout()
    plt.savefig("assets/class_distribution.png", dpi=150)
    plt.close()

    X = df["clean_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nTraining 3 models...")
    results_df, trained = evaluate_models(X_train, X_test, y_train, y_test)

    print("\nModel Comparison:")
    print(results_df.to_string(index=False))
    results_df.to_csv("model/model_comparison.csv", index=False)

    best_name = results_df.iloc[0]["Model"]
    best_pipeline = trained[best_name][0]
    best_predictions = trained[best_name][1]

    with open("model/best_model.pkl", "wb") as f:
        pickle.dump(best_pipeline, f)

    with open("model/model_info.txt", "w", encoding="utf-8") as f:
        f.write(f"Best model: {best_name}\n\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\nClassification report for best model:\n")
        f.write(classification_report(
            y_test, best_predictions,
            target_names=["Fake", "Real"]
        ))

    cm = confusion_matrix(y_test, best_predictions)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title(f"Confusion Matrix - {best_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["Fake", "Real"])
    plt.yticks([0, 1], ["Fake", "Real"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig("assets/confusion_matrix.png", dpi=150)
    plt.close()

    print(f"\nBest model: {best_name}")
    print("Saved: model/best_model.pkl")
    print("Saved: model/model_comparison.csv")
    print("Saved: assets/class_distribution.png")
    print("Saved: assets/confusion_matrix.png")


if __name__ == "__main__":
    main()
