import streamlit as st
import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
import tempfile

# ✅ Load the key securely
load_dotenv()

# Streamlit app config
st.set_page_config(page_title="Chat with Your Notes", layout="wide")
st.title("📄 Chat with Your Notes - Gen AI Project")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    # Load and split
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)

    if not docs:
        st.error("❌ Could not extract any text from the PDF.")
        st.stop()

    # Embeddings + Vector Store
    embeddings = OpenAIEmbeddings()
    try:
        vectorstore = FAISS.from_documents(docs, embeddings)
    except Exception as e:
        st.error("⚠️ OpenAI API error or rate limit.")
        st.stop()

    # Setup LLM QA Chain
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

    # User input
    question = st.text_input("Ask a question about the uploaded PDF:")
    if question:
        with st.spinner("Searching..."):
            try:
                result = qa_chain({"query": question})
                st.subheader("🔍 Answer:")
                st.write(result["result"])

                with st.expander("📄 Source Document Snippets"):
                    for i, doc in enumerate(result["source_documents"], 1):
                        st.markdown(f"**Snippet {i}:** {doc.page_content[:300]}...")
            except Exception:
                st.error("⚠️ Could not get a response. Try again.")
