import os
import json
import glob
from pathlib import Path
from openai import OpenAI
import logging
from ..utils import resize_images, load_config, get_openai_client, encode_image_to_base64
from dotenv import load_dotenv
load_dotenv()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

config = load_config("config_semantics.yaml")
RESIZED_DIR = Path(config["resized_dir"])
OUTPUT_PATH = Path(config["output_path"])
PROMPT_PATH = Path(config["prompt_path"])

def extract_semantics(client: OpenAI, image_paths: list[Path]) -> list[dict]:
    with open(PROMPT_PATH, "r") as f:
        prompt_text = f.read()

    results = []

    for path in image_paths:
        try:
            base64_img = encode_image_to_base64(path)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                    ]
                }]
            )
            description = response.choices[0].message.content.strip()
            results.append({
                "id": path.stem,
                "description": description
            })
            print(f"[Processed] {path.name}: {description}")

        except Exception as e:
            print(f"[Error] {path.name}: {e}")

    return results

def save_descriptions(descriptions: list[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(descriptions, f, indent=2)
    print(f"[Saved] Semantics to {output_path}")

if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment.")
    client = OpenAI(api_key=api_key)
    resized_images = sorted(RESIZED_DIR.glob("*.jpeg"))
    semantics = extract_semantics(client, resized_images)
    save_descriptions(semantics, OUTPUT_PATH)

