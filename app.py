import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Chat with Your Notes", layout="wide")
st.title("📚 Chat with Your Notes - Gen AI Project")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(docs, embeddings)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

    question = st.text_input("Ask a question about the uploaded PDF:")
    if question:
        with st.spinner("Finding answer..."):
            result = qa_chain({"query": question})
            st.subheader("Answer:")
            st.write(result["result"])

            with st.expander("Source Document"):
                for doc in result["source_documents"]:
                    st.markdown(f"**Page Content:** {doc.page_content[:300]}...")
