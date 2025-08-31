# =======================================================
# Train & Predict Resume Classification Pipeline
# =======================================================

import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import re
import PyPDF2

# -------------------------
# 1. Load raw resumes CSV
df = pd.read_csv("data/Resume.csv")  # Make sure path is correct
X = df['Resume']      # Column with raw resume text
y = df['Category']    # Column with labels

# -------------------------
# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# 3. Build pipeline (TF-IDF + Logistic Regression)
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
    ('clf', LogisticRegression(max_iter=1000, solver='liblinear', C=10))
])

# -------------------------
# 4. Train pipeline
pipeline.fit(X_train, y_train)
print("✅ Pipeline trained successfully")

# -------------------------
# 5. Evaluate model
y_pred = pipeline.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, y_pred)*100, 2), "%")
print("Classification Report:\n", classification_report(y_test, y_pred))

# -------------------------
# 6. Save pipeline
joblib.dump(pipeline, "models/new_resume_classifier_pipeline.pkl")
print("Pipeline saved with TF-IDF vectorizer included")

# -------------------------
# 7. Helper functions
def clean_text(text):
    """Lowercase, remove extra spaces & special characters"""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF"""
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + " "
    return text.strip()

def predict_resume(text, pipeline=pipeline):
    """Predict category and top-3 probabilities for raw text"""
    text_clean = clean_text(text)
    pred = pipeline.predict([text_clean])[0]
    probs = pipeline.predict_proba([text_clean])[0]
    top_idx = probs.argsort()[-3:][::-1]
    top3 = [(pipeline.classes_[i], probs[i]) for i in top_idx]
    return pred, top3

# -------------------------
# 8. Test sample prediction (raw text)
sample_text = "Experienced data scientist skilled in python, tensorflow, and keras with NLP experience."
pred, top3 = predict_resume(sample_text)
print("Raw text prediction")
print("Predicted:", pred)
print("Top-3:", [(c, f"{p:.2%}") for c,p in top3])
print("-"*60)

# -------------------------
# 9. Test sample prediction (PDF)
pdf_path = "data/sample_resume.pdf"  # Replace with your PDF path
pdf_text = extract_text_from_pdf(pdf_path)
pred_pdf, top3_pdf = predict_resume(pdf_text)
print("PDF prediction")
print("Predicted:", pred_pdf)
print("Top-3:", [(c, f"{p:.2%}") for c,p in top3_pdf])