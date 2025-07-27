import os
import torch
import nltk
import PyPDF2
import faiss
import numpy as np
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer

# Download required NLTK data
nltk.download("punkt")

# ===================== Model Loaders =====================

def load_t5_model():
    """
    Loads the FLAN-T5 model and tokenizer for answer generation.
    """
    model_name = "google/flan-t5-base"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.to("cpu")
    return tokenizer, model

def load_sentence_embedder():
    """
    Loads the SentenceTransformer model for semantic embeddings.
    """
    return SentenceTransformer("all-MiniLM-L6-v2")

# ===================== PDF Text Extraction =====================

def extract_text_chunks(pdf_folder, sentences_per_chunk=4):
    """
    Extracts text from all PDFs in the given folder, splits into chunks of N sentences.
    """
    corpus = []

    punkt_params = PunktParameters()
    punkt_params.abbrev_types = set(['dr', 'vs', 'mr', 'mrs'])
    tokenizer = PunktSentenceTokenizer(punkt_params)

    for filename in os.listdir(pdf_folder):
        if not filename.lower().endswith(".pdf"):
            continue

        file_path = os.path.join(pdf_folder, filename)
        try:
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                full_text = ""

                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        full_text += page_text

                sentences = tokenizer.tokenize(full_text)
                chunks = [" ".join(sentences[i:i + sentences_per_chunk])
                          for i in range(0, len(sentences), sentences_per_chunk)]
                corpus.extend(chunks)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
    return corpus

# ===================== Semantic Indexing =====================

def build_faiss_index(corpus, embedder):
    """
    Encodes the corpus using SentenceTransformer and builds a FAISS index.
    """
    embeddings = embedder.encode(corpus, convert_to_numpy=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])  # Cosine similarity
    index.add(embeddings)
    return index, embeddings

def retrieve_top_chunks(query, embedder, index, corpus, embeddings, top_k=3):
    """
    Retrieves top-k semantically similar chunks for a given query.
    """
    query_embedding = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    _, indices = index.search(query_embedding, top_k)
    return [corpus[i] for i in indices[0]]

# ===================== Answer Generation =====================

def generate_answer(question, tokenizer, model, context_chunks):
    """
    Generates an answer using FLAN-T5 based on the retrieved context.
    """
    context = " ".join(context_chunks)
    prompt = f"Answer the following question based only on the given context.\nQuestion: {question}\nContext: {context}"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, padding=True)
    outputs = model.generate(inputs["input_ids"], max_length=150, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ===================== Main CLI App =====================

def main():
    pdf_folder = "knowledge_base/"

    if not os.path.exists(pdf_folder) or not os.listdir(pdf_folder):
        print(f"Error: The folder '{pdf_folder}' does not exist or contains no PDFs.")
        return

    print("Loading models...")
    tokenizer, t5_model = load_t5_model()
    embedder = load_sentence_embedder()

    print("Extracting and indexing PDF content...")
    corpus = extract_text_chunks(pdf_folder)
    if not corpus:
        print("No valid text found in PDFs.")
        return

    index, embeddings = build_faiss_index(corpus, embedder)

    print("Ready. You can now ask questions based on the PDF content.")

    while True:
        question = input("\nEnter your question (or type 'exit' to quit): ").strip()
        if question.lower() == "exit":
            print("Exiting.")
            break

        top_chunks = retrieve_top_chunks(question, embedder, index, corpus, embeddings)

        # Show retrieved context
        print("\nRetrieved context chunks:")
        for i, chunk in enumerate(top_chunks):
            print(f"[{i+1}]: {chunk[:300]}...\n")

        answer = generate_answer(question, tokenizer, t5_model, top_chunks)
        print("Answer:", answer)

# ===================== Entry Point =====================

if __name__ == "__main__":
    main()
