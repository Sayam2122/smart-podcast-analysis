import os
import json
import numpy as np
import faiss
from typing import List, Dict, Any, Optional

class VectorStore:
    """
    Handles semantic search and metadata storage for podcast segments using FAISS.
    Stores all segment metadata for rich context display and analytics.
    """
    def __init__(self, index_dir: str, episode_id: str, embedding_model_name: str):
        self.index_dir = index_dir
        self.episode_id = episode_id
        self.embedding_model_name = embedding_model_name
        self.index_path = os.path.join(index_dir, f"{episode_id}.faiss")
        self.metadata_path = os.path.join(index_dir, f"{episode_id}_meta.json")
        self.embedding_model = None
        self.index = None
        self.metadata = []
        self._load()

    def _load(self):
        """
        Loads the FAISS index and metadata from disk if available.
        """
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)

    def is_built(self) -> bool:
        """Checks if the vector store is already built and loaded."""
        return self.index is not None and self.metadata

    def build(self, segments: List[Dict[str, Any]]):
        """
        Builds the FAISS index and metadata from a list of enriched segments.
        Args:
            segments: List of segment dicts from DataLoader.
        """
        if not segments:
            print("[VectorStore] No segments provided for building index.")
            return
        self.metadata = segments
        # Lazy import for memory efficiency
        from sentence_transformers import SentenceTransformer
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
        texts = [seg.get('text', '') for seg in segments]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        embedding_dim = embeddings.shape[1]
        cpu_index = faiss.IndexFlatL2(embedding_dim)
        self.index = faiss.IndexIDMap(cpu_index)
        ids = np.array(range(len(texts)))
        self.index.add_with_ids(embeddings, ids)
        self._save()

    def _save(self):
        """Saves the FAISS index and metadata to disk."""
        os.makedirs(self.index_dir, exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2)

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs a semantic search and returns top-k segments with relevance scores and metadata.
        Args:
            query: User's search query.
            k: Number of top results to return.
        Returns:
            List of segment dicts with added 'relevance_score' field.
        """
        if not self.is_built():
            print("[VectorStore] Vector store not built for this episode.")
            return []
        from sentence_transformers import SentenceTransformer
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
        query_emb = self.embedding_model.encode([query])
        distances, indices = self.index.search(query_emb, k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            result = dict(self.metadata[idx])
            result['relevance_score'] = float(dist)
            results.append(result)
        return results 