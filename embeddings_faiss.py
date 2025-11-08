# backend/embeddings_faiss.py
import os
import numpy as np
import faiss
import json
from sentence_transformers import SentenceTransformer

MODEL_NAME = os.getenv('HF_EMB_MODEL','all-MiniLM-L6-v2')

class FaissIndex:
    def __init__(self, model_name=MODEL_NAME, dim=None):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.dim = dim
        self.snippets = []

    def build(self, snippets):
        texts = [s['text'] for s in snippets]
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        # normalize
        faiss.normalize_L2(embeddings)
        self.dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)
        self.snippets = snippets
        return embeddings

    def save(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(out_dir, 'faiss.index'))
        with open(os.path.join(out_dir, 'snippets.json'), 'w', encoding='utf-8') as f:
            json.dump(self.snippets, f, ensure_ascii=False, indent=2)

    def load(self, out_dir):
        self.index = faiss.read_index(os.path.join(out_dir, 'faiss.index'))
        with open(os.path.join(out_dir, 'snippets.json'), 'r', encoding='utf-8') as f:
            self.snippets = json.load(f)

    def query(self, query_text, top_k=5):
        q_emb = self.model.encode([query_text], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.snippets):
                continue
            results.append({'score': float(score), 'snippet': self.snippets[idx]})
        return results
