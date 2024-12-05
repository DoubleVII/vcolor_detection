import utils
from typing import List, Dict
import os
import requests
from tqdm import tqdm


def download():

    save_path = "face_img"
    timeout = 5
    vinfo = utils.load_vinfo()

    vinfo = utils.prefilter(vinfo)

    print("begin download")

    success_count = 0

    for item in tqdm(vinfo):
        face_url = item["face"]
        file_name = os.path.basename(face_url)
        local_file_path = os.path.join(save_path, file_name)

        if os.path.exists(local_file_path):
            success_count += 1
            continue

        try:
            response = requests.get(face_url, timeout=timeout)
            if response.status_code == 200:
                with open(local_file_path, "wb") as file:
                    file.write(response.content)
                success_count += 1
            else:
                raise Exception("Failed to download image")
        except:
            print(f"Failed to download image: {face_url}")

    print(f"Successfully downloaded {success_count} images out of {len(vinfo)}")


if __name__ == "__main__":
    download()
