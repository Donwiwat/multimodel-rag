import textwrap
from pathlib import Path
from typing import Any

from core.loan import calculate_loan, parse_down_payment

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

    def _format_model_info(self, results: list[dict[str, Any]]) -> str:
        """Add the catalog facts for each distinct model among the results."""
        sections = []
        model_keys = dict.fromkeys(r["model_key"] for r in results if "model_key" in r)

        for key in model_keys:
            info = self.retriever.model_info(key)
            if not info:
                continue

            lines = [f"{info.get('make', '')} {info.get('model', key)}".strip()]
            if info.get("summary"):
                lines.append(info["summary"])
            if info.get("visual_cues"):
                lines.append("Identifying visual cues:")
                lines += [f"- {cue}" for cue in info["visual_cues"]]
            if info.get("specs"):
                lines.append("Specifications:")
                lines += [f"- {name}: {value}" for name, value in info["specs"].items()]
            sections.append("\n".join(lines))

        return "\n\n".join(sections)

    # Cosine-score cutoffs, measured on the GR Yaris data. Image queries:
    # GR Yaris ~0.85+, other cars ~0.71. Text queries: GR Yaris ~0.62+,
    # other cars ~0.56 or lower.
    IMAGE_MIN_SCORE = 0.78
    TEXT_MIN_SCORE = 0.59

    def _retrieve(
        self,
        path: Path | None,
        question: str,
        top_k: int,
        min_score: float | None,
    ) -> list[dict[str, Any]]:
        """Retrieve records by image, or by the question when there is no image."""
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"Image file not found at {path}")

        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        if path is not None:
            query_embedding = self.embedder.embed(str(path))
            default_min_score = self.IMAGE_MIN_SCORE
        else:
            query_embedding = self.embedder.embed_text(question)
            default_min_score = self.TEXT_MIN_SCORE
        if min_score is None:
            min_score = default_min_score

        retrieved_items = self.retriever.retrieve(query_embedding, k=top_k)

        # Drop weak matches so an unknown car is not described with another
        # model's specs.
        return [
            item for item in retrieved_items
            if item.get("score", 1.0) >= min_score
        ]

    def loan_quote(
        self,
        image_path: str | Path | None = None,
        question: str = "",
        price: float | None = None,
        down_payment: str | float = 0,
        annual_rate: float = 0.0,
        months: int = 60,
        method: str = "flat",
    ) -> dict[str, Any]:
        """
        Identify the car, then quote a loan on its price.

        The price comes from the model's info.json ("price" and "currency")
        unless one is passed in. The numbers are computed, not generated.
        """
        path = Path(image_path) if image_path else None
        retrieved_items = self._retrieve(path, question, top_k=5, min_score=None)
        model_key = next((r["model_key"] for r in retrieved_items if "model_key" in r), None)
        info = self.retriever.model_info(model_key) if model_key else {}
        name = f"{info.get('make', '')} {info.get('model', model_key or '')}".strip()

        if price is None:
            price = info.get("price")
        if price is None:
            if model_key is None:
                raise ValueError("The car was not recognised; pass a price to quote a loan.")
            raise ValueError(f"No price for {name} in info.json; pass a price to quote a loan.")

        quote = calculate_loan(
            price=float(price),
            down_payment=parse_down_payment(down_payment, float(price)),
            annual_rate=annual_rate,
            months=months,
            method=method,
        )

        return {
            "model_key": model_key,
            "model_name": name or None,
            "currency": info.get("currency", ""),
            "quote": quote,
            "retrieved_results": retrieved_items,
        }

    def query(
        self,
        image_path: str | Path | None = None,
        question: str = "Describe this car.",
        top_k: int = 5,
        min_score: float | None = None,
    ) -> dict[str, Any]:
        """
        Answer a question about a car, from an image, a text question, or both.

        With an image, retrieval uses the image; without one, it uses the question.
        """
        path = Path(image_path) if image_path else None
        retrieved_items = self._retrieve(path, question, top_k, min_score)

        context = self._format_context(retrieved_items)
        model_info = self._format_model_info(retrieved_items)
        if model_info:
            context = f"{context}\n\n{model_info}"

        source = "the supplied image and the" if path is not None else "the"

        # Dedent the template before filling it in, so multi-line context
        # does not break the indentation detection.
        prompt = textwrap.dedent("""            You are a car visual analysis assistant.

            Answer the user's question using {source}
            retrieved reference information.

            User question:
            {question}

            Retrieved car reference information:
            {context}

            Instructions:
            - Describe only features supported by the image or references.
            - Distinguish visible evidence from uncertain identification.
            - Do not claim an exact model year or trim without evidence.
            - If the references do not contain the answer, say so explicitly.
        """).format(source=source, question=question, context=context)

        if not retrieved_items:
            # Small VLMs ignore "say so if the references lack the answer" and
            # invent specs, so state the missing match explicitly.
            if path is not None:
                prompt += textwrap.dedent("""                    - This car does NOT match any vehicle in the reference library.
                      Start your answer by saying so. Describe only what is visible
                      in the image (body style, color, lights, wheels, badges).
                      Do NOT state engine, transmission, power, safety or other
                      specifications.
                """)
            else:
                prompt += textwrap.dedent("""                    - The question does NOT match any vehicle in the reference
                      library. Say so, and do NOT state any specifications.
                """)

        answer = self.inference.generate(
            image_paths=[str(path)] if path is not None else [],
            prompt=prompt
        )

        return {
            "image_path": str(path) if path is not None else None,
            "question": question,
            "answer": answer,
            "retrieved_results": retrieved_items
        }


if __name__ == "__main__":
    import argparse

    from core.embedder import CarImageEmbedder
    from core.indexer import INDEX_FILE, CarImageIndexer
    from core.inference import SwiftVLMEngine
    from core.loan import add_loan_arguments, format_quote
    from core.retriever import CarImageRetriever

    parser = argparse.ArgumentParser(description="Ask a question about a car, with or without an image.")
    parser.add_argument("question", nargs="?", default="What car is this? Describe it and its key specifications.")
    parser.add_argument("--image", help="Path to a car image to ask about")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rebuild", action="store_true", help="Re-index assets/ before querying")
    parser.add_argument("--loan", action="store_true", help="Identify the car and quote a loan instead of answering")
    add_loan_arguments(parser)
    args = parser.parse_args()

    if args.image is None and args.question == parser.get_default("question") and not (args.loan and args.price):
        parser.error("give a question, an --image, or both")

    index_dir = Path("data/index")
    embedder = CarImageEmbedder("./models/Qwen3-VL-Embedding-2B")

    if args.rebuild or not (index_dir / INDEX_FILE).is_file():
        indexer = CarImageIndexer(embedder, index_dir)
        count = indexer.build("assets")
        indexer.save()
        print(f"Indexed {count} record(s) into {index_dir}")

    pipeline = CarVisualRAGPipeline(
        embedder=embedder,
        retriever=CarImageRetriever(index_dir),
        # A loan quote is pure arithmetic, so skip loading the VLM.
        inference=None if args.loan else SwiftVLMEngine("./models/Qwen2-VL-2B-Instruct"),
    )

    if args.loan:
        try:
            loan = pipeline.loan_quote(
                args.image, args.question, price=args.price, down_payment=args.down,
                annual_rate=args.rate, months=args.months, method=args.method,
            )
        except ValueError as error:
            parser.error(str(error))
        print(f"Car: {loan['model_name'] or 'not recognised'}")
        print(format_quote(loan["quote"], loan["currency"]))
        raise SystemExit

    result = pipeline.query(args.image, args.question, top_k=args.top_k)
    print("Retrieved:")
    for item in result["retrieved_results"]:
        print(f"  {item['score']:.3f}  {item['kind']:<12}  {item['image_path']}")
    print("Answer:", result["answer"])
