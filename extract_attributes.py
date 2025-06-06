import os
import json
import glob
from pathlib import Path
from openai import OpenAI
import logging
from utils import load_config, encode_image_to_base64

from dotenv import load_dotenv
load_dotenv()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__) 

config = load_config("config_attributes.yaml")
RESIZED_DIR = Path(config["resized_dir"])
OUTPUT_PATH = Path(config["output_path"])
PROMPT_PATH = Path(config["prompt_path"])

def extract_attributes(client: OpenAI, image_paths: list[Path]) -> list[dict]:
    """
    Send images to OpenAI GPT-4o with a prompt to extract structured clothing attributes.

    Args:
        client (OpenAI): Initialized OpenAI client.
        image_paths (list[Path]): List of image file paths to process.

    Returns:
        list[dict]: List of dictionaries containing extracted attributes and image IDs.
    """
    attributes = []

    PROMPT_PATH = Path(__file__).parent / "prompt.txt"
    with open(PROMPT_PATH, "r") as f:
        prompt_text = f.read()

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
            content = response.choices[0].message.content.strip("```json\n").strip("```")
            data = json.loads(content)
            data["id"] = path.stem
            attributes.append(data)
            print(f"[Processed] {path.name}: {data}")
        except Exception as e:
            print(f"[Error] {path.name}: {e}")

    return attributes

def save_attributes(attributes: list[dict], output_path: Path):
    """
    Save the extracted attribute list to a JSON file.

    Args:
        attributes (list[dict]): List of attribute dictionaries to save.
        output_path (Path): File path to write the output JSON.
    """
    with open(output_path, "w") as f:
        json.dump(attributes, f, indent=2)
    print(f"[Saved] Attributes to {output_path}")

if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment.")
    client = OpenAI(api_key=api_key)
    resized_images = sorted(RESIZED_DIR.glob("*.jpeg"))
    attributes = extract_attributes(client, resized_images)
    save_attributes(attributes, OUTPUT_PATH)
