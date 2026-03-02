# RAG Website Q&A

End-to-end Retrieval-Augmented Generation pipeline that scrapes any website, builds a FAISS vector index, and answers questions grounded solely in the scraped content.

## Architecture

```
URL → Scraper → Clean Text → Chunking → Embeddings → FAISS
                                                        ↓
User Question → Similarity Search → Top-K Chunks → LLM → Answer
```

| Component | Tech |
|---|---|
| Scraping | `requests` + `BeautifulSoup4` |
| Chunking | `RecursiveCharacterTextSplitter` (1000 / 200 overlap) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (free, local) |
| Vector DB | FAISS (local, persisted to `faiss_index/`) |
| LLM | Groq `llama-3.1-8b-instant` (free API) |
| Frontend | Streamlit |

## Project Structure

```
8-RAG-Website-QA/
├── app.py              # Streamlit UI
├── rag_pipeline.py     # All logic: scraping, chunking, embedding, retrieval, LLM
├── rag_pipeline.ipynb  # Jupyter notebook version
├── requirements.txt
├── .env                # API keys (not committed)
└── README.md
```

## Setup

```bash
# 1. Create / activate a virtual environment
python -m venv .venv
source .venv/Scripts/activate   # Git Bash on Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your free Groq API key (https://console.groq.com/keys)
#    Edit .env and replace your_groq_api_key_here with a real key

# 4. Run
streamlit run app.py
```

## Usage

1. Paste a website URL in the sidebar and click **Process Website**.
2. Once processing completes, type a question in the main area.
3. Click **Get Answer** — the response is generated from the scraped content only.
4. Expand **Retrieved Context Chunks** to inspect what the model saw.

Each question is independent — no conversation memory is used.
