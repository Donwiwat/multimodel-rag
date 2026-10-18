import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pymupdf

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".jfif", ".png", ".webp", ".bmp"}
INDEX_FILE = "index.faiss"
METADATA_FILE = "metadata.json"
INFO_FILE = "info.json"
# 72 DPI keeps a catalog spread around 1 megapixel, which fits a 6GB GPU.
PAGE_DPI = 72


class CarImageIndexer:
    """Builds a FAISS index of car photos and catalog pages and saves it to disk.

    Expects a library laid out as <root>/<make>/<model>/ containing
    images/ (photos), catalog/ (PDFs) and an optional info.json.
    """

    def __init__(self, embedder: Any, index_dir: str | Path = "data/index") -> None:
        self.embedder = embedder
        self.index_dir = Path(index_dir)
        self.index: faiss.Index | None = None
        self.records: list[dict[str, Any]] = []
        self.models: dict[str, dict[str, Any]] = {}

    def add(self, image_path: str | Path, record: dict[str, Any]) -> None:
        """Embed one image and add it to the index with its metadata."""
        vector = np.asarray([self.embedder.embed(image_path)], dtype=np.float32)

        if self.index is None:
            # Embeddings are L2-normalized, so inner product == cosine similarity.
            self.index = faiss.IndexFlatIP(vector.shape[1])

        self.index.add(vector)
        self.records.append({"image_path": str(image_path), **record})

    def build(self, root: str | Path = "assets") -> int:
        """Index every <make>/<model> folder under root. Returns the record count."""
        model_dirs = sorted(p for p in Path(root).glob("*/*") if p.is_dir())
        if not model_dirs:
            raise FileNotFoundError(f"No <make>/<model> folders found in {root}")

        for model_dir in model_dirs:
            self._add_model(model_dir)

        return len(self.records)

    def _add_model(self, model_dir: Path) -> None:
        key = f"{model_dir.parent.name}/{model_dir.name}"
        info_path = model_dir / INFO_FILE
        info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.is_file() else {}
        name = " ".join(filter(None, [info.get("make"), info.get("model")])) or key
        self.models[key] = info

        for photo in sorted((model_dir / "images").glob("*")):
            if photo.suffix.lower() in IMAGE_EXTENSIONS:
                self.add(photo, {
                    "model_key": key,
                    "kind": "photo",
                    "description": f"Photo of a {name}.",
                })

        page_dir = self.index_dir / "pages"
        page_dir.mkdir(parents=True, exist_ok=True)
        page_notes = info.get("catalog_pages", {})

        for pdf in sorted((model_dir / "catalog").glob("*.pdf")):
            with pymupdf.open(pdf) as doc:
                for number, page in enumerate(doc, start=1):
                    page_path = page_dir / f"{model_dir.parent.name}_{model_dir.name}_{pdf.stem}_p{number}.png"
                    page.get_pixmap(dpi=PAGE_DPI).save(page_path)
                    note = page_notes.get(str(number), "")
                    self.add(page_path, {
                        "model_key": key,
                        "kind": "catalog_page",
                        "source": f"{pdf.name} page {number}",
                        "description": f"{name} catalog page {number}. {note}".strip(),
                    })

    def save(self) -> None:
        if self.index is None:
            raise RuntimeError("Nothing to save: the index is empty.")

        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_dir / INDEX_FILE))
        (self.index_dir / METADATA_FILE).write_text(
            json.dumps({"models": self.models, "records": self.records}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
