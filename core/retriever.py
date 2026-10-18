import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from core.indexer import INDEX_FILE, METADATA_FILE


class CarImageRetriever:
    """Finds the most similar indexed car images for a query embedding."""

    def __init__(self, index_dir: str | Path = "data/index") -> None:
        index_dir = Path(index_dir)
        index_path = index_dir / INDEX_FILE
        if not index_path.is_file():
            raise FileNotFoundError(
                f"No index at {index_path}. Build one with CarImageIndexer first."
            )

        self.index = faiss.read_index(str(index_path))
        metadata = json.loads((index_dir / METADATA_FILE).read_text(encoding="utf-8"))
        self.records: list[dict[str, Any]] = metadata["records"]
        self.models: dict[str, dict[str, Any]] = metadata["models"]

    def retrieve(self, query_embedding: list[float], k: int = 5) -> list[dict[str, Any]]:
        """Return up to k records, most similar first, each with a cosine score."""
        query = np.asarray([query_embedding], dtype=np.float32)
        k = min(k, self.index.ntotal)
        if k == 0:
            return []

        scores, ids = self.index.search(query, k)

        return [
            {**self.records[i], "score": float(score)}
            for score, i in zip(scores[0], ids[0])
            if i != -1
        ]

    def model_info(self, model_key: str) -> dict[str, Any]:
        """Return the info.json contents for a model, or {} if it has none."""
        return self.models.get(model_key, {})
