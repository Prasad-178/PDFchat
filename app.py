import streamlit as st
from PyPDF2 import PdfReader
import io
import base64
import os
from dotenv import load_dotenv
from Document import Document
from chain import Chain
import random
import time

load_dotenv()

# Function to save uploaded PDF file
def save_uploaded_file(uploaded_file):
    if not os.path.exists('pdf_files'):
        os.makedirs('pdf_files')
    file_path = os.path.join('pdf_files', uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    document_loader = Document(file_path)
    return document_loader

# Function to simulate LLM response
def response_generator(prompt, retriever, chain):
    response = chain.invoker(prompt, "session_1")
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

# Set page config
st.set_page_config(page_title="PDF Chat App", layout="wide")

# Apply dark mode and custom CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #1E1E1E;
        color: white;
    }
    .stTextInput > div > div > input {
        color: white;
    }
    .stButton > button {
        color: white;
        background-color: #4CAF50;
        border: none;
    }
    [data-testid="stChatMessage"] {
        background-color: #475063;
        margin-left: 0 !important;
        margin-right: 20% !important;
    }
    [data-testid="stChatMessage"][data-chatmessage-role="user"] {
        background-color: #2b313e;
        margin-left: 20% !important;
        margin-right: 0 !important;
    }
    </style>
    <script>
    const scrollToBottom = () => {
        const chatContainer = document.querySelector('.stChatMessageContainer');
        if (chatContainer) {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    };
    window.addEventListener('load', scrollToBottom);
    window.addEventListener('DOMContentLoaded', (event) => {
        const observer = new MutationObserver(scrollToBottom);
        observer.observe(document.body, { childList: true, subtree: true });
    });
    </script>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'pdf_uploaded' not in st.session_state:
    st.session_state['pdf_uploaded'] = False
if 'initial_message' not in st.session_state:
    st.session_state['initial_message'] = None
if 'pdf_name' not in st.session_state:
    st.session_state['pdf_name'] = None
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'home'

# Define pages
def home_page():
    st.title("PDF Chat App")
    
    uploaded_file = st.file_uploader("Upload your PDF file", type="pdf", key="pdf_uploader")
    initial_message = st.text_input("Enter your first message (optional)")
    
    if uploaded_file is not None:
        file_path = save_uploaded_file(uploaded_file)
        st.session_state['pdf_uploaded'] = True
        st.session_state['document_loader'] = extract_text_from_pdf(file_path)
        st.session_state['pdf_name'] = uploaded_file.name
        st.session_state['initial_message'] = initial_message
        st.session_state['current_page'] = 'chat'
        st.rerun()
    elif initial_message:
        st.session_state['pdf_uploaded'] = True
        st.session_state['initial_message'] = initial_message
        st.session_state['current_page'] = 'chat'
        st.rerun()

def chat_page():
    with st.sidebar:
        st.title('💬 PDFChat')
        if st.button("Back to Home"):
            st.session_state['pdf_uploaded'] = False
            st.session_state['pdf_name'] = None
            st.session_state['current_page'] = 'home'
            st.rerun()
        
        if st.session_state['pdf_name']:
            st.write(f"Current PDF: {st.session_state['pdf_name']}")
    
    st.title("Chat with your PDF")
    
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
        
    if 'document_loader' in st.session_state:
        retriever = st.session_state['document_loader'].retriever
    else:
        retriever = None
    
    if 'chain' not in st.session_state:
        st.session_state['chain'] = Chain(retriever, "session_1")
        
    chain = st.session_state['chain']
    
    # Display welcome message if it's the first message
    if not st.session_state['chat_history']:
        welcome_message = "Welcome to the PDF Chat App! I'm here to assist you with any questions about the uploaded PDF. How can I help you today?"
        st.chat_message("assistant").markdown(welcome_message)
        st.session_state['chat_history'].append({"role": "assistant", "content": welcome_message})
    
    # Display chat messages from history
    for message in st.session_state['chat_history']:
        with st.chat_message(message["role"]):
            content = message["content"].replace("\n", "\n ") if message["role"] == "assistant" else message["content"]
            st.markdown(content)

    # Process initial message if present
    if st.session_state['initial_message']:
        prompt = st.session_state['initial_message']
        st.session_state['initial_message'] = None  # Clear the initial message
        st.chat_message("user").markdown(prompt)
        st.session_state['chat_history'].append({"role": "user", "content": prompt})
        
        with st.chat_message("assistant"):
            response = st.write_stream(response_generator(prompt, retriever, chain))
        
        st.session_state['chat_history'].append({"role": "assistant", "content": response})

    # React to user input
    if prompt := st.chat_input("What is up?"):
        st.chat_message("user").markdown(prompt)
        st.session_state['chat_history'].append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            response = st.write_stream(response_generator(prompt, retriever, chain))
        
        st.session_state['chat_history'].append({"role": "assistant", "content": response})

# Main app logic
if st.session_state['current_page'] == 'chat' and st.session_state['pdf_uploaded']:
    chat_page()
else:
    home_page()