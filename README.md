# RAG CLI Assistant

A lightweight Retrieval-Augmented Generation (RAG) assistant built with the
Claude API and scikit-learn. Built to run comfortably on constrained
environments (developed on Termux, Android) with no vector database and no
heavy dependencies.

## Why this project

This project demonstrates the core RAG pipeline end to end:

1. **Synthetic knowledge base generation** -- FAQ/manual-style docs generated
   by the Claude API (`generate_corpus.py`)
2. **Retrieval** -- TF-IDF vectorization + cosine similarity to find the most
   relevant docs for a query (`retriever.py`)
3. **Augmented generation** -- retrieved docs injected into the Claude API
   prompt as grounding context (`generator.py`)
4. **CLI interface** tying it all together (`main.py`)

## Architecture

```
                 ┌────────────────────┐
                 │ generate_corpus.py │
                 │  (Claude API)      │
                 └─────────┬──────────┘
                           │ writes
                           ▼
                    data/*.txt (corpus)
                           │
                           ▼
   query ──────►  ┌────────────────┐
                   │  retriever.py  │  TF-IDF + cosine similarity
                   │  (scikit-learn)│  → top-k relevant chunks
                   └───────┬────────┘
                           │ context
                           ▼
                   ┌────────────────┐
                   │  generator.py  │  Claude API + grounded prompt
                   └───────┬────────┘
                           │
                           ▼
                     answer (CLI)
```

## Setup

```bash
pip install -r requirements.txt --break-system-packages   # Termux
# or: pip install -r requirements.txt                      # elsewhere

export ANTHROPIC_API_KEY=your_key_here
```

## Usage

```bash
# 1. Generate the synthetic knowledge base
python generate_corpus.py

# 2. Run the assistant
python main.py

# Compare against a no-retrieval baseline
python main.py --no-rag

# Retrieve more chunks per query
python main.py --top-k 5

# Sanity-check retrieval quality
python eval_retrieval.py
```

## Example session

```
RAG CLI Assistant -- mode: RAG
Type a question, or 'quit' to exit.

> How do I reset my password?

[retrieval] query: 'How do I reset my password?'
  doc_01.txt  (score=0.5123)
  doc_05.txt  (score=0.1284)
  doc_03.txt  (score=0.0871)

To reset your password, go to the login page and click "Forgot password"...
```

## Why TF-IDF instead of embeddings/a vector DB

For a small, static corpus (dozens to low hundreds of docs), TF-IDF + cosine
similarity is fast, has no external dependencies beyond scikit-learn, and
keeps the whole pipeline runnable on a phone via Termux. Swapping in
embeddings (e.g. `voyageai` or Claude embeddings) and a real vector store
would be a natural next step for a larger or semantically trickier corpus.

## Project structure

```
rag-cli-assistant/
├── generate_corpus.py   # synthetic doc generation (Claude API)
├── retriever.py          # TF-IDF retrieval
├── generator.py          # grounded generation (Claude API)
├── main.py                # CLI entrypoint
├── eval_retrieval.py     # small retrieval sanity-check eval
├── data/                  # generated corpus lives here
├── requirements.txt
└── README.md
```

## Built with

- [Anthropic Claude API](https://docs.claude.com)
- scikit-learn (TF-IDF, cosine similarity)
- Python standard library only otherwise

---
Built as part of a hands-on portfolio exploring Retrieval-Augmented
Generation patterns from Anthropic's "Building with Claude API" course.
