import os
import yaml
import logging
import base64
from pathlib import Path

def encode_image_to_base64(image_path: Path) -> str:
    """
    Encode an image file as a base64 string.

    Args:
        image_path (Path): Path to the image file.

    Returns:
        str: Base64-encoded image.
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)
