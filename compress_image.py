import os
from PIL import Image
import glob


def compress_image(img_path, max_size=200_000):
    with Image.open(img_path) as img:
        while os.path.getsize(img_path) > max_size:
            img = img.resize(
                (int(img.width * 0.9), int(img.height * 0.9)), Image.LANCZOS
            )
            img.save(img_path)


def scan_and_compress(data_path: str, max_size=200_000):
    files = glob.glob(os.path.join(data_path, "*"))
    for file in files:
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            if os.path.getsize(file) > max_size:
                print(f"Compressing: {os.path.basename(file)}")
                compress_image(file, max_size)


if __name__ == "__main__":
    from utils import CLI

    CLI(scan_and_compress)
