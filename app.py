import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
import tempfile
import os
from dotenv import load_dotenv

# Load .env (not required on Streamlit, but safe for local testing)
load_dotenv()

# Streamlit UI setup
st.set_page_config(page_title="Chat with Your Notes", layout="wide")
st.title("📄 Chat with Your Notes - Gen AI Project")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)

    if not docs:
        st.error("❌ Could not extract any text from the PDF. Try another file.")
        st.stop()

    # Create embeddings & FAISS vectorstore
    embeddings = OpenAIEmbeddings()

    try:
        vectorstore = FAISS.from_documents(docs, embeddings)
    except Exception as e:
        st.error("⚠️ OpenAI API rate limit reached or usage exceeded. Try a smaller file or wait and retry.")
        st.stop()

    # Load LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

    # Ask a question
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
            except Exception as e:
                st.error("⚠️ Could not get a response. Please try again.")
