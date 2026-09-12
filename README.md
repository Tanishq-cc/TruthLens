# TruthLens 📰
## Fake News Classifier using Machine Learning

TruthLens is an NLP-based supervised machine-learning project that classifies news articles as **Fake** or **Real**.

It follows the GDG JIIT 128 AI/ML task requirements:
- Text cleaning
- EDA
- TF-IDF
- Three ML models
- Accuracy and F1-score comparison

## Dataset

We use the **WELFake** dataset. The original dataset contains 72,134 accessible news articles, with 35,028 real and 37,106 fake articles. Its CSV contains `title`, `text`, and `label` fields; **0 = fake and 1 = real**.

Dataset source:
https://zenodo.org/records/4561253

A Kaggle mirror is also available:
https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification

Download `WELFake_Dataset.csv` and place it at:

```text
TruthLens/data/WELFake_Dataset.csv
```

## Project Structure

```text
TruthLens/
├── data/
│   └── WELFake_Dataset.csv
├── model/
├── assets/
├── train_model.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup on Mac

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python3 train_model.py
```

The script cleans the text, performs a class-distribution EDA plot, creates an 80/20 stratified split, applies TF-IDF, trains three models, compares metrics, and saves the best model.

Models:
1. Logistic Regression
2. Multinomial Naive Bayes
3. Linear SVM

Metrics:
- Accuracy
- Precision
- Recall
- F1-score

## Run the Web App

```bash
streamlit run app.py
```

## Interview Explanation

> "I built a fake news classifier using NLP and supervised machine learning. First, I cleaned the news text and performed EDA. I used TF-IDF to convert text into numerical features. Then I trained Logistic Regression, Naive Bayes and Linear SVM and compared their Accuracy and F1-score. Finally, I saved the best-performing model and connected it to a Streamlit interface for Real/Fake prediction."

## Important Limitation

The classifier learns patterns from its training data. It does not independently fact-check an article, so a prediction is not proof that a story is true or false.

## GitHub

Do not upload:
- `venv/`
- secrets/API keys
- large raw datasets unless their license permits redistribution

The `.gitignore` is already included.


## Streamlit Features

The app now includes:
- Separate title and article inputs
- Real/Fake example buttons for demos
- Prediction result and model confidence indicator
- Model performance dashboard
- Accuracy/F1 comparison for all three models
- Class distribution and confusion-matrix visualizations
- "How It Works" section for interview/demo explanation
- Clear limitation note explaining that the classifier is not a fact-checker

## Run the app

```bash
source venv/bin/activate
streamlit run app.py
```
