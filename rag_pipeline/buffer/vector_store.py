# /podcast_rag_project/vector_store.py

import os
import json
import numpy as np
import faiss
# Lazy import to avoid memory issues: from sentence_transformers import SentenceTransformer
import logging

class VectorStore:
    """
    Manages a discrete FAISS vector index and associated metadata for a single episode.
    This approach ensures that each episode's data is isolated and efficiently managed.
    """

    def __init__(self, vector_store_dir, episode_id, embedding_model_name):
        """
        Initializes the VectorStore for a specific episode.
        
        Args:
            vector_store_dir (str): The base directory where all vector stores are saved.
            episode_id (str): The unique identifier for the episode (e.g., the folder name).
            embedding_model_name (str): The name of the sentence-transformer model to use.
        """
        # Each episode gets its own set of uniquely named files in the vector_store directory
        self.index_path = os.path.join(vector_store_dir, f"{episode_id}.faiss")
        self.metadata_path = os.path.join(vector_store_dir, f"{episode_id}_meta.json")
        self.embedding_model_name = embedding_model_name
        self.embedding_model = None  # Lazy initialization
        self.index = None
        self.metadata = []
        self._load()

    def _load(self):
        """
        Loads the FAISS index and metadata from disk if they already exist for the episode.
        This prevents re-calculating embeddings on subsequent runs, saving significant time.
        """
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            logging.info(f"Loading existing vector store for episode '{os.path.basename(self.index_path)}' from disk.")
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        else:
            logging.info(f"No existing vector store found for this episode. It will need to be built.")

    def is_built(self):
        """Checks if the vector store is already built and loaded for this episode."""
        return self.index is not None and self.metadata

    def build(self, enriched_data):
        """
        Builds the FAISS index and metadata from scratch using the enriched segment data
        for the episode and saves them to disk.
        
        Args:
            enriched_data (list): A list of dictionary objects, where each object is a
                                  fully enriched segment from the DataLoader.
        """
        if not enriched_data:
            logging.error("Cannot build vector store: No data provided.")
            return

        logging.info(f"Building vector store for episode...")
        self.metadata = enriched_data
        
        # Initialize embedding model only when needed
        if self.embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logging.info(f"Loading embedding model: {self.embedding_model_name}")
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
            except (ImportError, MemoryError) as e:
                logging.error(f"Failed to load embedding model: {e}")
                raise Exception(f"Cannot build vector store without embedding model: {e}")
        
        # Extract the text from each segment to be converted into a vector embedding
        texts = [segment.get('text', '') for segment in self.metadata]
        
        logging.info(f"Generating embeddings for {len(texts)} texts. This may take a moment...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        embedding_dim = embeddings.shape[1]
        # Using IndexFlatL2 for standard Euclidean distance search
        cpu_index = faiss.IndexFlatL2(embedding_dim)
        # Using IndexIDMap to map the vector's position back to our original metadata index
        self.index = faiss.IndexIDMap(cpu_index)
        ids = np.array(range(len(texts)))
        self.index.add_with_ids(embeddings, ids)

        self._save()

    def _save(self):
        """Saves the episode's FAISS index and metadata to disk for future use."""
        logging.info(f"Saving FAISS index to {self.index_path}")
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)

        logging.info(f"Saving metadata to {self.metadata_path}")
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2)

    def search(self, query_text, k=5):
        """
        Performs a semantic search on the episode's vector store.
        
        Args:
            query_text (str): The user's question or search term.
            k (int): The number of top results to return.
            
        Returns:
            list: A list of the top k matching enriched segment objects.
        """
        if not self.is_built():
            logging.error("Search failed: Vector store has not been built for this episode.")
            return []

        # Initialize embedding model only when needed for search
        if self.embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logging.info(f"Loading embedding model for search: {self.embedding_model_name}")
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
            except (ImportError, MemoryError) as e:
                logging.error(f"Failed to load embedding model for search: {e}")
                return []

        query_embedding = self.embedding_model.encode([query_text])
        # The search returns distances and the indices of the matching vectors
        distances, indices = self.index.search(query_embedding, k)
        
        # Retrieve the full metadata for each of the top k results
        results = [self.metadata[i] for i in indices[0] if i < len(self.metadata)]
        logging.info(f"Found {len(results)} relevant segments from vector search.")
        return results
