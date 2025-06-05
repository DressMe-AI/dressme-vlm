from PIL import Image
from pathlib import Path

DATA_DIR = "./data/images"
RESIZED_DIR = "./data/resized"
IMG_SIZE = [256, 256]

def resize_images(source_dir: Path, dest_dir: Path,
                  size: tuple[int, int]) -> list[Path]:
    """
    Resize all images in a directory to a fixed size, centered on a white canvas.

    Args:
        source_dir (Path): Directory containing original images.
        dest_dir (Path): Output directory for resized images.
        size (tuple[int, int]): Target width and height.

    Returns:
        list[Path]: Paths to resized images.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    resized_paths = []

    for img_name in os.listdir(source_dir):
        try:
            src_path = source_dir / img_name
            dest_path = dest_dir / img_name

            img = Image.open(src_path).convert("RGB")
            img.thumbnail(size, Image.LANCZOS)
            new_img = Image.new("RGB", size, (255, 255, 255))
            offset = ((size[0] - img.width) // 2, (size[1] - img.height) // 2)
            new_img.paste(img, offset)
            new_img.save(dest_path, "JPEG", quality=85)
            resized_paths.append(dest_path)
            logger.info(f"Resized {img_name}")

        except Exception as e:
            logger.error(f"Could not process {img_name}: {e}")

    return resized_paths

if __name__ == "__main__":
    resized_images = resize_images(DATA_DIR, RESIZED_DIR, IMG_SIZE)
