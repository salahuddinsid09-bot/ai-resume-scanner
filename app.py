import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Extract text from PDF
def extract_text(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Load job description
def load_jd():
    with open("job_description.txt", "r") as f:
        return f.read()

st.title("AI Resume Screening System")

uploaded_files = st.file_uploader("Upload resumes", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    jd = load_jd()

    resumes = []
    names = []

    for file in uploaded_files:
        text = extract_text(file)
        resumes.append(text)
        names.append(file.name)

    documents = [jd] + resumes

    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform(documents)

    scores = cosine_similarity(vectors[0:1], vectors[1:])[0]

    st.subheader("Ranking")

    results = list(zip(names, scores))
    results.sort(key=lambda x: x[1], reverse=True)

    for name, score in results:
        st.write(f"{name} - {round(score*100,2)}%")