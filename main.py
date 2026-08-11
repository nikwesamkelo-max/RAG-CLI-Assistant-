"""
main.py

CLI entrypoint for the RAG assistant.

Usage:
    export ANTHROPIC_API_KEY=your_key_here
    python main.py
    python main.py --no-rag     # compare answers without retrieval
    python main.py --top-k 5    # retrieve more chunks per query
"""

import os
import argparse
import anthropic

from retriever import Retriever
from generator import answer_with_context, answer_without_context


def parse_args():
    parser = argparse.ArgumentParser(description="RAG CLI Assistant")
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Skip retrieval; answer from the model alone (comparison mode).",
    )
    parser.add_argument(
        "--top-k", type=int, default=3, help="Number of chunks to retrieve per query."
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Force rebuild of the TF-IDF index instead of using the cache.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("Set ANTHROPIC_API_KEY before running this script.")
    client = anthropic.Anthropic(api_key=api_key)

    retriever = None
    if not args.no_rag:
        retriever = Retriever(docs_dir="data")
        retriever.build(use_cache=not args.rebuild_index)

    mode = "no-RAG (baseline)" if args.no_rag else "RAG"
    print(f"RAG CLI Assistant -- mode: {mode}")
    print("Type a question, or 'quit' to exit.\n")

    while True:
        query = input("> ").strip()
        if query.lower() in ("quit", "exit"):
            break
        if not query:
            continue

        if args.no_rag:
            answer = answer_without_context(client, query)
        else:
            retrieved = retriever.explain_retrieval(query, top_k=args.top_k)
            answer = answer_with_context(client, query, retrieved)

        print(f"\n{answer}\n")


if __name__ == "__main__":
    main()
