import os
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import numpy as np


def extract_text_from_pdfs(pdf_folder):
    documents = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            filepath = os.path.join(pdf_folder, filename)
            with open(filepath, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                documents.append(text)
    return documents


def create_tfidf_index(documents):
    vectorizer = TfidfVectorizer(stop_words='english')
    doc_vectors = vectorizer.fit_transform(documents)
    index = faiss.IndexFlatL2(doc_vectors.shape[1])
    index.add(doc_vectors.toarray())
    return index, vectorizer


def retrieve_top_k(query, index, vectorizer, documents, k=3):
    query_vector = vectorizer.transform([query]).toarray()
    distances, indices = index.search(query_vector, k)
    results = [documents[idx] for idx in indices[0]]
    return results
