import streamlit as st
import joblib
import re
import PyPDF2

# -------------------------
# 1. Load trained pipeline
pipeline = joblib.load("models/new_resume_classifier_pipeline.pkl")

# -------------------------
# 2. Helper functions
def clean_text(text):
    """Lowercase, remove special chars, extra spaces"""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF"""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + " "
    return text.strip()

def predict_resume(text):
    """Return predicted category and top-3 probabilities"""
    text_clean = clean_text(text)
    pred = pipeline.predict([text_clean])[0]
    probs = pipeline.predict_proba([text_clean])[0]
    top_idx = probs.argsort()[-3:][::-1]
    top3 = [(pipeline.classes_[i], probs[i]) for i in top_idx]
    return pred, top3

# -------------------------
# 3. Streamlit UI
st.title("📄 Resume Classifier")
st.write("Upload a PDF or type/paste resume text to classify it into a job category.")

# Option: PDF upload
uploaded_file = st.file_uploader("Upload PDF resume", type=["pdf"])

# Option: Text input
text_input = st.text_area("Or paste resume text here")

# Predict button
if st.button("Predict"):
    if uploaded_file is not None:
        # Extract text from PDF
        resume_text = extract_text_from_pdf(uploaded_file)
        st.write("**Extracted Resume Text:**")
        st.write(resume_text[:1000] + "..." if len(resume_text) > 1000 else resume_text)
    elif text_input.strip() != "":
        resume_text = text_input
    else:
        st.warning("Please upload a PDF or enter some text.")
        st.stop()
    
    # Prediction
    pred, top3 = predict_resume(resume_text)
    st.success(f"**Predicted Category:** {pred}")
    st.write("**Top-3 Predictions:**")
    for category, prob in top3:
        st.write(f"{category}: {prob:.2%}")