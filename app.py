import streamlit as st
import joblib
import nltk
import PyPDF2
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
import os

# Ensure necessary resources are downloaded
nltk.download('punkt')

# Load model and vectorizer
model = joblib.load("ai_human_classifier_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# --- Utility Functions ---
def read_txt(file):
    return file.read().decode("utf-8")

def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def read_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def preprocess(text):
    tokens = word_tokenize(text.lower())
    return " ".join(tokens)

# --- Streamlit App ---
st.set_page_config(page_title="Human vs AI Text Detector", layout="centered")
st.title("🧠 Human vs 🤖 AI Text Classifier")

st.write("Upload a file or paste your text below to classify it as **AI-generated** or **Human-written**.")

# --- Text Input Area ---
input_text = st.text_area("✍️ Enter your text here:", height=200)

# --- File Upload ---
uploaded_file = st.file_uploader("📂 Or upload a text file (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"])

if uploaded_file is not None:
    if uploaded_file.type == "text/plain":
        input_text = read_txt(uploaded_file)
    elif uploaded_file.type == "application/pdf":
        input_text = read_pdf(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        input_text = read_docx(uploaded_file)
    st.success("✅ File uploaded successfully!")

# --- Prediction Button ---
if st.button("🔍 Classify Text"):
    if input_text.strip() == "":
        st.warning("Please enter or upload some text first.")
    else:
        cleaned_text = preprocess(input_text)
        vect_text = vectorizer.transform([cleaned_text])
        prediction = model.predict(vect_text)[0]
        proba = model.predict_proba(vect_text)[0]

        label = "🤖 AI-Generated" if prediction == 1 else "👤 Human-Written"
        confidence = round(max(proba) * 100, 2)

        st.markdown(f"### 📢 Prediction: {label}")
        st.markdown(f"**Confidence:** {confidence}%")
