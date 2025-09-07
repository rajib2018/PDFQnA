import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint
import sys
import traceback # Import the traceback module

# Set the Hugging Face Hub API token as an environment variable
# Replace "YOUR_HF_API_TOKEN" with your actual token
# You can obtain a token from your Hugging Face account settings
# It's recommended to use Streamlit secrets for actual deployment
# Ensure this token is set before the app logic
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

def process_pdf(uploaded_file):
    """Processes the uploaded PDF, creates a vector store."""
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            # Load and split the PDF
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            text_chunks = text_splitter.split_documents(documents)

            # Create embeddings and vector store
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vector_store = FAISS.from_documents(text_chunks, embeddings)

            return vector_store
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            st.error(traceback.format_exc()) # Print traceback
            return None
        finally:
            # Clean up the temporary file
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
    return None

# Set up the Streamlit interface
st.title("PDF Question Answering System")
st.write("Upload a PDF and ask a question about its content.")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Use Streamlit's session state to store the vector store
if 'vector_store' not in st.session_state:
    st.session_state['vector_store'] = None

if uploaded_file is not None and st.session_state['vector_store'] is None:
    st.success("PDF uploaded successfully! Processing...")
    st.session_state['vector_store'] = process_pdf(uploaded_file)

if st.session_state['vector_store'] is not None:
    question = st.text_input("Ask a question about the PDF:")

    if question:
        # Instantiate the LLM (only once per session if possible, or here)
        repo_id = "google/flan-t5-large"
        try:
            llm = HuggingFaceEndpoint(
                repo_id=repo_id, temperature=0.5
            )

            # Create a RetrievalQA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm, retriever=st.session_state['vector_store'].as_retriever()
            )

            # Get the answer
            with st.spinner("Getting answer..."):
                answer = qa_chain.invoke(question)
            st.write("Answer:")
            st.write(answer['result'])
        except Exception as e:
            st.error(f"An error occurred while getting the answer: {e}")
            st.error(traceback.format_exc()) # Print traceback

else:
    st.info("Please upload a PDF to get started.")

# Add code to run the streamlit app
if __name__ == "__main__":
    import subprocess
    # Save the script to a temporary file
    script_content = """
import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint
import subprocess
import sys
import traceback # Import the traceback module

# Set the Hugging Face Hub API token as an environment variable
# Replace "YOUR_HF_API_TOKEN" with your actual token
# You can obtain a token from your Hugging Face account settings
# It's recommended to use Streamlit secrets for actual deployment
# Ensure this token is set before the app logic
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "YOUR_HF_API_TOKEN"

def process_pdf(uploaded_file):
    \"\"\"Processes the uploaded PDF, creates a vector store.\"\"\"
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", mode="wb") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            # Load and split the PDF
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            text_chunks = text_splitter.split_documents(documents)

            # Create embeddings and vector store
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vector_store = FAISS.from_documents(text_chunks, embeddings)

            return vector_store
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            st.error(traceback.format_exc()) # Print traceback
            return None
        finally:
            # Clean up the temporary file
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
    return None

# Set up the Streamlit interface
st.title("PDF Question Answering System")
st.write("Upload a PDF and ask a question about its content.")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Use Streamlit's session state to store the vector store
if 'vector_store' not in st.session_state:
    st.session_state['vector_store'] = None

if uploaded_file is not None and st.session_state['vector_store'] is None:
    st.success("PDF uploaded successfully! Processing...")
    st.session_state['vector_store'] = process_pdf(uploaded_file)

if st.session_state['vector_store'] is not None:
    question = st.text_input("Ask a question about the PDF:")

    if question:
        # Instantiate the LLM (only once per session if possible, or here)
        repo_id = "google/flan-t5-large"
        try:
            llm = HuggingFaceEndpoint(
                repo_id=repo_id, temperature=0.5
            )

            # Create a RetrievalQA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm, retriever=st.session_state['vector_store'].as_retriever()
            )

            # Get the answer
            with st.spinner("Getting answer..."):
                answer = qa_chain.invoke(question)
            st.write("Answer:")
            st.write(answer['result'])
        except Exception as e:
            st.error(f"An error occurred while getting the answer: {e}")
            st.error(traceback.format_exc()) # Print traceback

else:
    st.info("Please upload a PDF to get started.")
"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as f:
        f.write(script_content.encode())
        script_path = f.name

    # Run the streamlit app using subprocess
    try:
        process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", script_path])
        process.wait()
    except Exception as e:
        st.error(f"Error running Streamlit app: {e}")
    finally:
        # Clean up the temporary script file
        if os.path.exists(script_path):
            os.remove(script_path)
