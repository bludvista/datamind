# 🧠 DataMind — Agentic AI Pipeline

Automated HR Attrition Analysis powered by Multi-Agent AI, RAG & LLM.

## Features
- 6 autonomous AI agents working in sequence
- RAG system using HuggingFace Sentence Transformers + FAISS
- LLM-generated HR insight reports via OpenRouter API
- Interactive Streamlit UI with CSV upload
- Auto-generated charts and downloadable HTML report

## Tech Stack
Python | Streamlit | OpenRouter API | RAG | HuggingFace | FAISS | Pandas | Matplotlib

## How to Run
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Add your `OPENROUTER_API_KEY` to `.env`
4. Run: `streamlit run app.py`
