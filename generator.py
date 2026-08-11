"""
generator.py

Takes a user query + retrieved context chunks and calls the Claude API
to produce a grounded answer. This is the "augmented generation" half
of RAG -- the prompt template here is the part worth pointing to in a
portfolio writeup.
"""

import os
import anthropic

MODEL = "claude-sonnet-4-5"

SYSTEM_PROMPT = """You are a support assistant for the product Flowdesk.
Answer the user's question using ONLY the context provided below. If the
context does not contain the answer, say you don't have that information
rather than guessing.

Be concise and direct."""


def build_context_block(retrieved) -> str:
    """retrieved: list of (doc_name, text, score) tuples from Retriever."""
    parts = []
    for name, text, score in retrieved:
        parts.append(f"[{name}] (relevance={score:.3f})\n{text}")
    return "\n\n".join(parts)


def answer_with_context(client: anthropic.Anthropic, query: str, retrieved) -> str:
    context = build_context_block(retrieved)
    prompt = f"""Context:
{context}

Question: {query}"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def answer_without_context(client: anthropic.Anthropic, query: str) -> str:
    """Baseline call with no retrieval -- useful for the --no-rag comparison
    flag to demonstrate what RAG actually buys you."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=500,
        system="You are a helpful assistant for the product Flowdesk.",
        messages=[{"role": "user", "content": query}],
    )
    return response.content[0].text.strip()
