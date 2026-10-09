"""MCP docs search and retrieval server using TF-IDF."""

import json
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

mcp = FastMCP("DocsServer")

# Load documents corpus
DOCS_PATH = Path("data/kb/documents.jsonl")
documents = []
if DOCS_PATH.exists():
    with open(DOCS_PATH, "r") as f:
        for line in f:
            documents.append(json.loads(line))

corpus = [d["text"] for d in documents] if documents else [""]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus) if documents else None


@mcp.tool()
def search_documents(query: str, top_k: int = 3) -> str:
    """Search knowledge base documents using TF-IDF cosine similarity."""
    if not documents or tfidf_matrix is None:
        return json.dumps([])

    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = sims.argsort()[::-1][:top_k]

    results = []
    for idx in top_indices:
        if sims[idx] > 0.0:
            doc = documents[idx]
            results.append({
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "section": doc["section"],
                "text": doc["text"],
                "score": float(sims[idx]),
            })
    return json.dumps(results)


@mcp.tool()
def get_document(doc_id: str) -> str:
    """Retrieve a specific document by its doc_id."""
    for doc in documents:
        if doc["doc_id"] == doc_id:
            return json.dumps(doc)
    return json.dumps({"error": f"Document {doc_id} not found"})


if __name__ == "__main__":
    mcp.run()
