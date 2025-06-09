import os
import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from utils import encode_image_to_base64
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

load_dotenv()

config = load_config("config_semantics.yaml")
RESIZED_DIR = Path(config["resized_dir"])
PROMPT_PATH = Path(config["prompt_path"])

GOLD_PATH = Path("data/actual_desc.json")

def load_gold_descriptions():
    with open(GOLD_PATH, "r") as f:
        return json.load(f)  # { "top_1": "gold desc", ... }

def load_prompt():
    with open(PROMPT_PATH, "r") as f:
        return f.read().strip()

def generate_descriptions(client: OpenAI, image_paths: list[Path], prompt_text: str):
    results = []
    for path in image_paths:
        img_id = path.stem
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
        gen_text = response.choices[0].message.content.strip()
        results.append((img_id, gen_text))
        print(f"[Generated] {img_id}: {gen_text}")
    return results

def evaluate_outputs(generated, golds):
    rouge = Rouge()
    results = []

    for img_id, gen_text in generated:
        gold_text = golds.get(img_id)
        if not gold_text:
            print(f"[Warning] No gold description for {img_id}")
            continue

        bleu = sentence_bleu([gold_text.split()], gen_text.split(), weights=(0.5, 0.5))
        rouge_scores = rouge.get_scores(gen_text, gold_text)[0]

        result = {
            "id": img_id,
            "generated": gen_text,
            "gold": gold_text,
            "bleu": round(bleu, 4),
            "rouge_l": round(rouge_scores["rouge-l"]["f"], 4)
        }
        results.append(result)

    return results

def print_results(results):
    for r in results:
        print(f"\n[ID: {r['id']}]")
        print(f"- BLEU: {r['bleu']}")
        print(f"- ROUGE-L: {r['rouge_l']}")
        print(f"- Gen:  {r['generated']}")
        print(f"- Gold: {r['gold']}")

if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    prompt_text = load_prompt()
    image_paths = sorted(TEST_IMAGE_DIR.glob("*.jpeg"))
    golds = load_gold_descriptions()

    generated = generate_descriptions(client, image_paths, prompt_text)
    results = evaluate_outputs(generated, golds)
    print_results(results)
