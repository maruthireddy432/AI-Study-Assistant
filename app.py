import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()

# --- Page and UI Configuration ---
st.set_page_config(
    page_title="EduBot: Your Personal Study Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 EduBot")
st.markdown("Your AI-powered study assistant. Upload a document and start asking questions!")

# --- Sidebar for File Upload and Instructions ---
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader(
        "Upload a PDF or TXT file:",
        type=["pdf", "txt"],
        help="The chatbot will answer questions based only on the content of this document."
    )
    
    st.info("💡 **How it works:**\n\n"
            "1. Upload your study material (PDF or text).\n"
            "2. EduBot processes the content and builds a knowledge base.\n"
            "3. Ask questions and get answers, without any external knowledge!\n\n"
            "This app leverages LangChain for the RAG pipeline and Groq for blazing-fast LLM inference."
    )

    st.markdown("---")
    st.header("API Key Configuration")
    st.markdown("A Groq API key is required to use the LLM.")
    groq_api_key = st.text_input(
        "Enter your Groq API Key:",
        type="password",
        help="Get your key from https://console.groq.com/keys"
    )

    # Note for deployment on Streamlit Cloud
    st.markdown(
        "**For Streamlit Cloud deployment**, add your `GROQ_API_KEY` to the app's secrets section."
    )

# --- Session State Initialization ---
# This is crucial for maintaining conversational history and the vector store across interactions.
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "processing_status" not in st.session_state:
    st.session_state.processing_status = ""

# --- Core RAG Pipeline Functions ---

def load_and_split_documents(uploaded_file):
    """
    Loads and splits a document into manageable text chunks.
    Handles both PDF and TXT file types.
    """
    st.session_state.processing_status = "⏳ Processing document..."
    
    # Create a temporary file to save the uploaded content
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.type.split('/')[-1]}") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        file_path = temp_file.name

    try:
        if uploaded_file.type == "application/pdf":
            loader = PyPDFLoader(file_path)
        else: # Assumes text/plain
            loader = TextLoader(file_path)

        # Load the document and split into chunks
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        text_chunks = text_splitter.split_documents(docs)
        return text_chunks

    finally:
        os.remove(file_path)
        st.session_state.processing_status = ""

def create_vector_store(text_chunks):
    """
    Generates embeddings and creates a FAISS vector store from the text chunks.
    """
    st.session_state.processing_status = "⏳ Building knowledge base..."
    # Use a simple, locally-runnable embedding model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(text_chunks, embeddings)
    st.session_state.processing_status = ""
    return vector_store
    
def get_answer_generator(stream):
    """
    A generator to extract only the 'answer' from the LangChain stream output.
    """
    for chunk in stream:
        if 'answer' in chunk:
            yield chunk['answer']


# --- Document Processing Flow ---
if uploaded_file and st.session_state.vector_store is None:
    if not groq_api_key:
        st.warning("Please provide a Groq API Key to proceed.")
    else:
        st.session_state.vector_store = None
        st.session_state.messages = []
        with st.spinner("Processing document..."):
            text_chunks = load_and_split_documents(uploaded_file)
            st.session_state.vector_store = create_vector_store(text_chunks)
        st.success("Document processed successfully! You can now ask questions.")

# --- Main Chat Interface ---

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Handle user input
if prompt := st.chat_input("Ask a question about the document..."):
    # Check for prerequisites
    if not groq_api_key:
        st.error("Please enter your Groq API key in the sidebar.")
        st.stop()
    if st.session_state.vector_store is None:
        st.error("Please upload a document first.")
        st.stop()
    
    # Add user message to chat history and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Start the AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Set up the Groq LLM and the RAG chain
            llm = ChatGroq(
                groq_api_key=groq_api_key, 
                model_name="openai/gpt-oss-20b", # You can also try "llama3-70b-8192"
                temperature=0.5
            )
            
            retriever = st.session_state.vector_store.as_retriever()
            
            # The system prompt is critical for preventing hallucination.
            # It explicitly instructs the model to only use the provided context.
            system_prompt = (
                "You are an educational assistant. Your sole purpose is to answer questions based strictly on the provided context."
                "Your tone should be clear, factual, and direct. Do not use any external knowledge."
                "If the answer cannot be found in the context, you must respond with: "
                "'I couldn't find this information in the uploaded material.'\n\n"
                "Context: {context}"
            )
            
            prompt_template = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("human", "{input}"),
                ]
            )

            # Create the chain using LangChain Expression Language (LCEL)
            document_chain = create_stuff_documents_chain(llm, prompt_template)
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            # Use the .stream() method to get a generator for streaming output
            stream_response = retrieval_chain.stream(
                {"input": prompt},
                config={"callbacks": None}
            )

            # Pass the LangChain stream to our custom generator to filter it for text only
            clean_stream = get_answer_generator(stream_response)
            
            # The `st.write_stream` function consumes the clean generator
            full_response = st.write_stream(clean_stream)

            # Store the final AI response in the session state
            st.session_state.messages.append({"role": "assistant", "content": full_response})
