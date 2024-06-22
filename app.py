import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

from flask import Flask, send_from_directory
from werkzeug.utils import secure_filename
from threading import Thread

from dotenv import load_dotenv
load_dotenv()

st.set_page_config(
    page_title="AI-DOCTOR Agent",
    page_icon="🧑‍⚕️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Load groq API Keys.
groq_api_key = "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"

st.title("Medical Report Summarization 🧑‍⚕️")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    You are an advanced AI agent designed to process and summarize medical reports, as well as provide useful insights based on the information provided. Your task involves:

Summarization:

Concisely summarize the key findings, diagnoses, treatments, and recommendations outlined in the medical report.
Highlight any significant medical history, lab results, imaging findings, and physical examination details.
Insight Extraction:

Answer specific questions related to the patient's condition, prognosis, and treatment options based on the report.
Provide explanations for medical terms and conditions mentioned in the report.
Suggest possible next steps or further tests if the information in the report indicates a need for them.
Identify any potential red flags or urgent issues that require immediate attention.

Example Medical Report:


Tasks:


    
    <context>
    {context}
    <context>
    Question:
    Summarize the medical report provided above.

Answer the following questions based on the report:

What is the primary diagnosis?
Are there any secondary conditions or complications mentioned?
What treatment plan has been recommended?
Are there any significant lab results or imaging findings?
What follow-up actions are advised?
Are there any potential risks or concerns highlighted in the report?
Explain any medical terms or conditions mentioned in the report in simple terms.

Suggest any additional tests or consultations if necessary.

Identify and elaborate on any urgent issues that need immediate attention. 

What are some insights you can carry out from . give me precautions and health care advice. use emojis to look attractive
    write in structured and readable format
    """
)

# Streamlit file uploader widget
uploaded_file = st.file_uploader("Upload your medical report PDF file", type="pdf")

# Set up a directory to save the uploaded files
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to run Flask server
def run_flask():
    app = Flask(__name__)

    @app.route('/uploads/<filename>')
    def uploaded_file(filename):
        return send_from_directory(UPLOAD_FOLDER, filename)

    app.run(port=8000)

# Start Flask server in a separate thread
if 'flask_thread' not in st.session_state:
    flask_thread = Thread(target=run_flask)
    flask_thread.start()
    st.session_state['flask_thread'] = flask_thread

# Initialize session state variables if they don't exist
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = HuggingFaceEmbeddings()
if 'text_splitter' not in st.session_state:
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

if uploaded_file is not None:
    # Secure the file name
    filename = secure_filename(uploaded_file.name)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    
    # Save the uploaded file to the specified file path
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    file_url = f"http://localhost:8000/uploads/{filename}"
    
    # Update session state with the file path
    st.session_state.file_path = file_path
    
    # Load documents and create vectors
    st.session_state.loader = PyPDFLoader(st.session_state.file_path)
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
    
    prompt1 = """
    What are some insights you can carry out from the PDF . give me precautions and health care advice. use emojis to look attractive
    write in structured and readable format
    """
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({'input': prompt1})
    st.write(response['answer'])

    st.download_button(
        label="Download Your Medical Summary",
        data=response['answer'],
        file_name="Medical_Report_Summary.txt",
        mime="text/plain",
    )
else:
    st.write("Please upload a medical report PDF file to get started.")
