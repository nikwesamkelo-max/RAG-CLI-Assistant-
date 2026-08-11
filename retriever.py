"""
retriever.py

Lightweight retrieval using TF-IDF + cosine similarity (scikit-learn).
No vector DB needed -- fine for small corpora and light on memory, which
matters when running on Termux/mobile.

The Retriever caches its fitted vectorizer + matrix with joblib so repeat
runs don't re-vectorize the corpus every time.
"""

import os
import glob
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CACHE_PATH = "data/.retriever_cache.joblib"


class Retriever:
    def __init__(self, docs_dir: str = "data"):
        self.docs_dir = docs_dir
        self.doc_names = []
        self.doc_texts = []
        self.vectorizer = None
        self.doc_matrix = None

    def _load_docs(self):
        paths = sorted(glob.glob(os.path.join(self.docs_dir, "*.txt")))
        if not paths:
            raise FileNotFoundError(
                f"No .txt docs found in {self.docs_dir}/. "
                "Run generate_corpus.py first."
            )
        for path in paths:
            with open(path, "r", encoding="utf-8") as f:
                self.doc_texts.append(f.read())
                self.doc_names.append(os.path.basename(path))

    def build(self, use_cache: bool = True):
        """Fit TF-IDF over the corpus, or load from cache if available."""
        if use_cache and os.path.exists(CACHE_PATH):
            cached = joblib.load(CACHE_PATH)
            self.doc_names = cached["doc_names"]
            self.doc_texts = cached["doc_texts"]
            self.vectorizer = cached["vectorizer"]
            self.doc_matrix = cached["doc_matrix"]
            return

        self._load_docs()
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_matrix = self.vectorizer.fit_transform(self.doc_texts)

        os.makedirs(self.docs_dir, exist_ok=True)
        joblib.dump(
            {
                "doc_names": self.doc_names,
                "doc_texts": self.doc_texts,
                "vectorizer": self.vectorizer,
                "doc_matrix": self.doc_matrix,
            },
            CACHE_PATH,
        )

    def retrieve(self, query: str, top_k: int = 3):
        """Return top_k (doc_name, text, score) tuples for a query."""
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.doc_matrix)[0]
        ranked = sorted(
            zip(self.doc_names, self.doc_texts, scores),
            key=lambda x: x[2],
            reverse=True,
        )
        return ranked[:top_k]

    def explain_retrieval(self, query: str, top_k: int = 3):
        """Print retrieval results with similarity scores -- makes the
        'black box' visible for demos/interviews."""
        results = self.retrieve(query, top_k=top_k)
        print(f"\n[retrieval] query: {query!r}")
        for name, _, score in results:
            print(f"  {name}  (score={score:.4f})")
        return results
