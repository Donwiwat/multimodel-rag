from aliyunsdkcore.vendored.requests.packages.urllib3 import response
from pathlib import Path

import torch
import torch.nn.functional as F
from swift import InferRequest, TransformersEngine

class CarImageEmbedder:
    def __init__(
        self, model_name: str = "Qwen/Qwen3-VL-Embedding-2B",
        device_map: str = "auto"
    ) -> None:
        """Embeds car images using Qwen2-VL-2B-Instruct."""
        self.model_name = model_name
        self.device_map = device_map
        self.engine = TransformersEngine(
            model=model_name,
            task_type="embeddings",
            device_map=device_map,
            max_batch_size=1
        )



    @torch.inference_mode()
    def embed(self, image_path: str | Path) -> list[float]:
        """Generate embedding vector for a single image."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found at {image_path}")
        request = InferRequest(
            messages = [
                {
                    "role": "user",
                    "content": (
                        "<image>Represent this car image for"
                        "visual similarity retrieval."
                    ),
                }
            ],
            images = [str(path.resolve())]
        )

        responses = self.engine.infer([request])
        if not responses:
            raise RuntimeError("Embeding did not return results")

        response_data = responses[0].data

        if not response_data:
            raise RuntimeError(
                "The embedding response contains no vector."
            )

        embedding = response_data[0]["embedding"]

        if embedding is None or len(embedding) == 0:
            raise RuntimeError(
                "The embedding model returned an empty vector."
            )
        
        vector = torch.as_tensor(
            embedding,
            dtype=torch.float32,
        ).flatten()

        if not torch.isfinite(vector).all():
            raise RuntimeError(
                "The embedding contains NaN or infinite values."
            )

        # Normalize for cosine-similarity search with FAISS.
        norm = torch.linalg.vector_norm(vector)

        if norm.item() == 0:
            raise RuntimeError(
                "The embedding has zero magnitude."
            )

        vector = F.normalize(vector, p=2, dim=0)

        return vector.cpu().tolist()


if __name__ == "__main__":
    model_name = "./models/Qwen3-VL-Embedding-2B"
    embedder = CarImageEmbedder(model_name)

    vector = embedder.embed("assets/car.jpg")

    print("Embedding dimension:", len(vector))
    print("First five values:", vector[:5])
    print("Vector norm:", sum(x * x for x in vector) ** 0.5)
        