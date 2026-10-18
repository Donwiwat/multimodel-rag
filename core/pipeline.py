from matplotlib.style import context
from pathlib import Path
from typing import Any

class CarVisualRAGPipeline:
    """Orchestrate the end-to-end car image RAG flow."""
    def __init__(self,
        embedder: Any,
        retriever: Any,
        inference: Any,
        indexer: Any | None = None
    ) -> None:
        self.embedder = embedder
        self.retriever = retriever
        self.inference = inference
        self.indexer = indexer

    @staticmethod
    def _format_context(results: Any) -> str:
        """Convert retrieved car records into prompt-friendly text."""
        if not results:
            return "No relevant records were found in the database."

        if isinstance(results, str):
            return results

        if isinstance(results, dict):
            results = [results]

        lines = []

        for index, result in enumerate(results, start=1):
            if isinstance(result, dict):
                description = (
                    result.get("description")
                    or result.get("text")
                    or result.get("content")
                    or str(result)
                )
            else:
                description = str(result)

            lines.append(f"{index}. {description}")

        return "\n".join(lines)
        
    def query(
        self,
        image_path: str | Path,
        question: str = "Describe this car.",
        top_k: int = 5,
    ) -> dict[str, Any]:
        
        """
        Retrieve relevant car images using the visual embedder and vector index.
        """
        path = Path(image_path)

        
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found at {image_path}")
        
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        #Generate image embedding for the query car image.
        query_embedding = self.embedder.embed(str(path))

        #Use the embedding to find the most similar images.
        retrieved_items = self.retriever.retrieve(query_embedding, k=top_k)
        
        context = self._format_context(retrieved_items)

        prompt = f"""
            You are a car visual analysis assistant.

            Answer the user's question using the suppiled image and the 
            retrieved reference information

            User question:
            {question}
            
            Retrieved car reference information:
            {context}

            Instructions:
            - Describe only features supported by the image or references
            - Distinguish visible evident from uncertain identification.
            - Do not claim an exact model year or trim without evidence.
            - If the reference do not contain the answer, say so explicitly.
        """
        answer  = self.inference.generate(
            image_paths=[str(Path)],
            prompt=prompt
        )

        return {
            "image_path": str(path),
            "question": question,
            "answer": answer,
            "retrieved_results": retrieved_items
        }