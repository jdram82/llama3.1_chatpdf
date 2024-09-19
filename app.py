import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from pdfminer.utils import open_filename
from pillow_heif import register_heif_opener

# load the environment variables
load_dotenv()

working_dir = os.path.dirname(os.path.abspath(__file__))

def load_document(file_path):
    try:
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        return documents
    except Exception as e:
        st.error(f"Failed to load the document: {str(e)}")
        return None

def setup_vectorstore(documents):
    if documents is None:
        return None
    
    embeddings = HuggingFaceEmbeddings()
    text_splitter = CharacterTextSplitter(
        separator="\n",  # Corrected separator
        chunk_size=1000,
        chunk_overlap=200
    )
    doc_chunks = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(doc_chunks, embeddings)
    return vectorstore

def create_chain(vectorstore):
    llm = ChatGroq(
        model="llama-3.1-70b-versatile",
        temperature=0
    )
    retriever = vectorstore.as_retriever()
    memory = ConversationBufferMemory(
        llm=llm,
        output_key="answer",
        memory_key="chat_history",
        return_messages=True
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        chain_type="map_reduce",
        memory=memory,
        verbose=True
    )
    return chain

st.set_page_config(
    page_title="Chat with Doc",
    page_icon="📄",
    layout="centered"
)

st.title("🦙 Chat with Doc - LLAMA 3.1")

# Initialize chat history in Streamlit session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader(label="Upload your PDF file", type=["pdf"])

if uploaded_file:
    # Save the uploaded file
    file_path = os.path.join(working_dir, uploaded_file.name)  # Better file path handling
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if "vectorstore" not in st.session_state:
        documents = load_document(file_path)
        if documents:
            st.session_state.vectorstore = setup_vectorstore(documents)

    if "conversation_chain" not in st.session_state:
        if st.session_state.vectorstore:
            st.session_state.conversation_chain = create_chain(st.session_state.vectorstore)

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input field
user_input = st.chat_input("Ask Llama...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Display user input
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        response = st.session_state.conversation_chain({"question": user_input})
        assistant_response = response["answer"]
        st.markdown(assistant_response)
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
