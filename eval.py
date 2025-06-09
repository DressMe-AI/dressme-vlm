import os
import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from utils import encode_image_to_base64, load_config
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

load_dotenv()

# Load config
config = load_config("config_semantics.yaml")
TEST_IMG_DIR = Path(config["resized_dir"])
PROMPT_PATH = Path(config["prompt_path"])
GOLD_PATH = Path("data/actual_desc.json")

# Select only these 6 images
SELECTED_IDS = {"top_9", "top_20", "top_25", "bottom_14", "bottom_5", "bottom_11"}

def load_gold_descriptions():
    with open(GOLD_PATH, "r") as f:
        return json.load(f)  # { "top_9": "...", ... }

def load_prompt():
    with open(PROMPT_PATH, "r") as f:
        return f.read().strip()

def generate_descriptions(client: OpenAI, image_paths: list[Path], prompt_text: str):
    results = []
    for path in image_paths:
        img_id = path.stem
        if img_id not in SELECTED_IDS:
            continue
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
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    results = []

    for img_id, gen_text in generated:
        gold_text = golds.get(img_id)
        if not gold_text:
            print(f"[Warning] No gold description for {img_id}")
            continue

        bleu = sentence_bleu([gold_text.split()], gen_text.split(), weights=(0.5, 0.5))
        scores = scorer.score(gold_text, gen_text)

        result = {
            "id": img_id,
            "generated": gen_text,
            "gold": gold_text,
            "bleu": round(bleu, 4),
            "rouge1": round(scores["rouge1"].fmeasure, 4),
            "rouge2": round(scores["rouge2"].fmeasure, 4),
            "rougeL": round(scores["rougeL"].fmeasure, 4)
        }
        results.append(result)

    return results

def print_results(results):
    for r in results:
        print(f"\n[ID: {r['id']}]")
        print(f"- BLEU: {r['bleu']}")
        print(f"- ROUGE-1: {r['rouge1']}")
        print(f"- ROUGE-2: {r['rouge2']}")
        print(f"- ROUGE-L: {r['rougeL']}")
        print(f"- Gen:  {r['generated']}")
        print(f"- Gold: {r['gold']}")

if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    prompt_text = load_prompt()
    image_paths = sorted(TEST_IMG_DIR.glob("*.jpeg"))
    golds = load_gold_descriptions()

    generated = generate_descriptions(client, image_paths, prompt_text)
    results = evaluate_outputs(generated, golds)
    print_results(results)
