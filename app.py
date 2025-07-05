import streamlit as st
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
import tempfile

# ✅ Manually set the OpenAI API key (for now)
os.environ["OPENAI_API_KEY"] = "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"

# Streamlit config
st.set_page_config(page_title="Chat with Your Notes", layout="wide")
st.title("📄 Chat with Your Notes - Gen AI Project")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents =
