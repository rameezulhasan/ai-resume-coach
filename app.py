"""
AI Career Coach - Resume Evaluation + Chat (Streamlit version)

What it does:
1. User uploads a resume PDF.
2. We extract the text and ask Groq (via LangChain) to summarize/evaluate it.
3. We save the resume text into a Chroma vector store (embeddings via local Ollama).
4. User can then chat about the resume. Memory is handled by LangChain itself
   using InMemoryChatMessageHistory + RunnableWithMessageHistory, so the
   chain automatically remembers everything said earlier in the chat.

Run with: streamlit run app.py

Note: Ollama must be running locally with the nomic-embed-text model pulled:
    ollama pull nomic-embed-text
"""

import os

import streamlit as st
from dotenv import load_dotenv
import PyPDF2

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever


# ---------------------------------------------------------
# Setup
# ---------------------------------------------------------

load_dotenv()  # reads the .env file

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")  # good general purpose Groq model

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Please add it to your .env file.")
    st.stop()

CHROMA_DIR = "chroma_db"
SESSION_ID = "resume_chat"  # single-user app, so one fixed session id is enough

# LLM (Groq, via LangChain)
llm = ChatGroq(
    model=GROQ_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0.3,
)

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    st.error("HF_TOKEN not found.")
    st.stop()

embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    task="feature-extraction",
    huggingfacehub_api_token=HF_TOKEN,
)

# Text splitter used before saving resume text into the vector store
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len,
)

# Embeddings via local Ollama (nomic-embed-text) - no API key needed


# ---------------------------------------------------------
# Prompt + chain for resume summary (ChatPromptTemplate + LCEL)
# ---------------------------------------------------------

resume_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an AI Career Coach. Given a candidate's resume, write a clear "
     "summary covering: Career Objective, Skills and Expertise, Professional "
     "Experience, Educational Background, and Notable Achievements. Keep it "
     "well-structured, concise, and professional."),
    ("human", "Here is the resume:\n\n{resume}"),
])

# LCEL chain: prompt -> llm -> parse to plain text
resume_chain = resume_prompt | llm | StrOutputParser()


# ---------------------------------------------------------
# Conversational RAG chain (with memory via RunnableWithMessageHistory)
# ---------------------------------------------------------

# Step 1: given chat history + a new question, rewrite the question so it
# makes sense on its own (e.g. "what about my last job?" -> a full question)
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Given the chat history and the latest user question, rewrite the "
     "question as a standalone question that can be understood without the "
     "chat history. Do NOT answer the question, just reformulate it if "
     "needed, otherwise return it as it is."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Step 2: answer the (rewritten) question using the retrieved resume context
qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an AI Career Coach chatting with a user about their resume. "
     "Answer using only the resume context below. If the answer is not in "
     "the context, say you don't have enough information.\n\nContext:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


# In-memory store that holds chat history objects, keyed by session_id.
# This dict lives inside Streamlit's session_state so it survives reruns.
def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in st.session_state.chat_store:
        st.session_state.chat_store[session_id] = InMemoryChatMessageHistory()
    return st.session_state.chat_store[session_id]


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def extract_text_from_pdf(uploaded_file):
    """Read an uploaded PDF file and return all its text as one string."""
    text = ""
    reader = PyPDF2.PdfReader(uploaded_file)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def build_vectorstore(resume_text):
    """Split resume text into chunks and save it as a Chroma vector store."""
    chunks = text_splitter.split_text(resume_text)

    # Fresh collection each time a new resume is uploaded
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name="resume_collection",
    )
    return vectorstore


def get_conversational_rag_chain():
    """Build the retrieval chain and wrap it with automatic memory handling."""
    db = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name="resume_collection",
    )
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # Retriever that first looks at chat history to understand the real question
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)

    # Combines retrieved chunks + chat history to produce the final answer
    document_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, document_chain)

    # Wrap the chain so LangChain automatically loads/saves chat history
    # for us, instead of us passing chat_history manually every time.
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational_rag_chain


# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------

st.set_page_config(page_title="AI Resume Coach", page_icon="📄")

st.title("AI Resume Coach")
st.write("Upload your resume as a PDF to get an AI-generated evaluation, then chat about it.")

# Keep the resume analysis + LangChain's chat history store in session state
# so they don't disappear every time Streamlit reruns the script
if "resume_analysis" not in st.session_state:
    st.session_state.resume_analysis = None

if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False

if "chat_store" not in st.session_state:
    st.session_state.chat_store = {}  # session_id -> InMemoryChatMessageHistory

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    if st.button("Evaluate Resume"):
        with st.spinner("Reading and analyzing your resume..."):
            resume_text = extract_text_from_pdf(uploaded_file)
            build_vectorstore(resume_text)
            st.session_state.resume_analysis = resume_chain.invoke({"resume": resume_text})
            st.session_state.vectorstore_ready = True
            st.session_state.chat_store = {}  # start a fresh chat memory for the new resume

if st.session_state.resume_analysis:
    st.subheader("Resume Analysis Results")
    st.write(st.session_state.resume_analysis)

st.divider()

# --------------------- Chat section (memory handled by LangChain) ---------------------

st.subheader("Chat About Your Resume")

if not st.session_state.vectorstore_ready:
    st.info("Upload and evaluate a resume first to enable the chat.")
else:
    # Show all previous messages saved in LangChain's chat history for this session
    history = get_session_history(SESSION_ID)
    for msg in history.messages:
        role = "user" if msg.type == "human" else "assistant"
        with st.chat_message(role):
            st.write(msg.content)

    # Chat input box (stays at the bottom, like a normal chat app)
    user_input = st.chat_input("Ask something about your resume...")

    if user_input:
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                conversational_rag_chain = get_conversational_rag_chain()

                # We only pass the new question. RunnableWithMessageHistory
                # automatically loads past messages before the call and
                # saves the new question + answer back into memory after.
                result = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": SESSION_ID}},
                )
                answer = result["answer"]
                st.write(answer)