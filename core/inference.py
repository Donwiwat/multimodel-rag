from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from swift import TransformersEngine, InferRequest

class SwiftVLMEngine:
    def __init__(self, model_path: str, device_map: str = "auto"):
        """Initialzes local Qwen2-VL-2B-Instruct model using ms-swift library"""
        self.model_path = model_path
        self.device_map = device_map
        self.engine = TransformersEngine(
            model=model_path,
            device_map=device_map,
        )

    def generate(self, image_paths: List[str], prompt: str) -> str:
        """Runs multimodal inference given image inputs and textual prompt."""    
        request = InferRequest(
         messages=[
            {"role": "user",
             "content": [
                {"image": img} for img in image_paths
             ]+[{"text": prompt}]
             }   
         ], 
        )
        response = self.engine.inference([request])
        return response[0].choices[0].message.content

if __name__ == "__main__":
    engine = SwiftVLMEngine(
        "./models/Qwen/Qwen2-VL-2B-Instruct",
        device_map="auto",
    )

    print(
        engine.generate(
            ["assets/car.jpg"],
            "Describe this car and its visible features.",
        )
    )