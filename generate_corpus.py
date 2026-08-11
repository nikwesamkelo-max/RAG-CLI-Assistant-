"""
generate_corpus.py

Generates a small synthetic knowledge base (FAQs + manual snippets) using the
Claude API, and saves each doc as a separate .txt file in data/.

This showcases prompt design for structured synthetic data generation --
one of the RAG pipeline stages worth demoing in interviews.

Usage:
    export ANTHROPIC_API_KEY=your_key_here
    python generate_corpus.py
"""

import os
import json
import anthropic

MODEL = "claude-sonnet-4-5"
OUTPUT_DIR = "data"

# Edit this list to change what your synthetic knowledge base is "about".
# Right now it's a fictional SaaS product's support docs -- swap the topic
# to whatever domain you want your demo to live in.
TOPICS = [
    "How to reset your password",
    "How to upgrade from Free to Pro plan",
    "How to invite team members to a workspace",
    "How to export data to CSV",
    "How to set up two-factor authentication",
    "How to cancel a subscription",
    "Troubleshooting failed payments",
    "How to integrate with Slack",
    "API rate limits and how to handle them",
    "How to delete your account and data",
]

SYSTEM_PROMPT = """You write concise FAQ/manual entries for a fictional SaaS
product called "Flowdesk". Each entry should read like real product
documentation: 3-6 sentences, practical, no fluff, no markdown headers.
Return ONLY the entry text, nothing else."""


def generate_doc(client: anthropic.Anthropic, topic: str) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Write the FAQ/manual entry for: {topic}"}],
    )
    return response.content[0].text.strip()


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("Set ANTHROPIC_API_KEY before running this script.")

    client = anthropic.Anthropic(api_key=api_key)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    manifest = []
    for i, topic in enumerate(TOPICS, start=1):
        print(f"[{i}/{len(TOPICS)}] Generating: {topic}")
        text = generate_doc(client, topic)
        filename = f"doc_{i:02d}.txt"
        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)
        manifest.append({"file": filename, "topic": topic})

    with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. {len(TOPICS)} docs written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
