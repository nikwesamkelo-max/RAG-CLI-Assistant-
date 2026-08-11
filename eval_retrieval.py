"""
eval_retrieval.py

Tiny sanity-check eval: a handful of test queries paired with the doc
we'd expect retrieval to surface. Not a rigorous benchmark -- just enough
to show you're thinking about retrieval quality, not only "does it run".

Usage:
    python eval_retrieval.py
"""

from retriever import Retriever

# Map query -> substring expected to appear in the top-1 retrieved doc's
# source topic (check data/manifest.json after generating your corpus and
# adjust these to match your actual generated topics).
TEST_CASES = [
    ("I forgot my password, how do I get back in?", "doc_01.txt"),
    ("How do I get more people onto my team's workspace?", "doc_03.txt"),
    ("My card got declined, what do I do?", "doc_07.txt"),
    ("I want to hook this up to Slack", "doc_08.txt"),
    ("How do I permanently remove my account?", "doc_10.txt"),
]


def main():
    retriever = Retriever(docs_dir="data")
    retriever.build()

    correct = 0
    for query, expected_doc in TEST_CASES:
        results = retriever.retrieve(query, top_k=1)
        top_doc = results[0][0]
        hit = top_doc == expected_doc
        correct += hit
        status = "PASS" if hit else "FAIL"
        print(f"[{status}] query={query!r} -> got={top_doc}, expected={expected_doc}")

    print(f"\n{correct}/{len(TEST_CASES)} top-1 retrievals matched expectations.")


if __name__ == "__main__":
    main()
