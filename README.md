# AI Resume Coach

A simple Streamlit app that evaluates your resume using an LLM (via Groq)
and then lets you chat about it. The chat has memory, so the LLM remembers
what was asked earlier in the same conversation.

## What it does

1. Upload your resume as a PDF.
2. The app extracts the text and asks the LLM (via LangChain + Groq) to
   write a structured evaluation: Career Objective, Skills, Experience,
   Education, and Achievements.
3. The resume text is also saved into a Chroma vector store (embeddings
   generated locally by Ollama's `nomic-embed-text` model).
4. You can then chat about the resume. The chain uses conversational RAG
   (retrieval + memory), so follow-up questions like "what about that?"
   are understood using the earlier messages.

## Tech used

- **Streamlit** — UI
- **LangChain** (LCEL: `prompt | llm | parser`) — chains
- **Groq** (`ChatGroq`) — the LLM
- **ChatPromptTemplate** — prompts
- **Chroma** — vector store
- **Ollama (`nomic-embed-text`)** — embeddings, runs locally
- **RunnableWithMessageHistory` + `InMemoryChatMessageHistory`** — chat memory

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running, with the embedding
  model pulled:
  ```bash
  ollama pull nomic-embed-text
  ```
- A free Groq API key: https://console.groq.com/keys

## Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/rameezulhasan/ai-resume-coach.git
   cd ai-resume-coach
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root with:
   ```
   GROQ_API_KEY=your_real_groq_api_key
   GROQ_MODEL=openai/gpt-oss-120b
   ```
   (`GROQ_MODEL` is optional — it defaults to `llama-3.3-70b-versatile` if
   you leave it out, but `openai/gpt-oss-120b` is currently a cheaper,
   larger-context option on Groq.)

5. Make sure Ollama is running in the background (it usually starts
   automatically after install; otherwise run `ollama serve`).

6. Run the app:
   ```bash
   streamlit run app.py
   ```

7. Open the URL Streamlit prints in your terminal (usually
   http://localhost:8501).

## Project structure

```
ai-resume-coach/
├── app.py              # everything: Streamlit UI + LangChain chains
├── requirements.txt
├── .env                 # you create this yourself (not committed to git)
├── chroma_db/            # created automatically after your first upload
└── README.md
```

## Notes

- Each time you upload and evaluate a new resume, the Chroma collection and
  chat memory are reset, so the chat only ever talks about the most
  recently uploaded resume.
- `.env` holds your real API key and should never be committed to GitHub —
  add it to `.gitignore` if you push this repo publicly.