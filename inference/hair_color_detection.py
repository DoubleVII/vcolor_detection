import json
import utils
import os
import inference.run_generation as run_generation


prompt = utils.load_config()["vcolor_detection_prompt"]

color_map = utils.load_config()["color_map"]


def prepare_messages(vinfo: list, img_cache_path: str = None) -> list:
    messages_list = []
    for item in vinfo:
        face_url = item["face"]
        if img_cache_path is not None:
            file_name = os.path.basename(face_url)
            local_file_path = os.path.join(img_cache_path, file_name)
            if os.path.exists(local_file_path):
                face_url = local_file_path

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": face_url,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        messages_list.append(messages)
    return messages_list


def post_process(output_text: str) -> str:
    color_names = color_map.keys()
    for color_name in color_names:
        if color_name in output_text:
            return color_map[color_name]
    return None


def run(
    data_path: str,
    output_path: str,
    model_path: str = "Qwen/Qwen2-VL-7B-Instruct",
    img_cache_path: str = None,
    prefilter: bool = True,
    batch_size: int = 1,
    pred_on_face_check: bool = True,
):
    assert data_path.endswith(".json"), "data_path must be a json file"
    assert output_path.endswith(".json"), "output_path must be a json file"

    vinfo = utils.load_vinfo(data_path)

    if prefilter:
        vinfo = utils.prefilter(vinfo)

    vinfo_dict = {item["mid"]: item for item in vinfo}

    if pred_on_face_check:
        vinfo_to_pred = [item for item in vinfo if item.get("face_check", False)]
    else:
        vinfo_to_pred = vinfo

    model, processor = utils.load_model_and_processor(model_path)

    messages_list = prepare_messages(vinfo_to_pred, img_cache_path)

    output_texts = run_generation.run(messages_list, batch_size, model, processor)

    for item, output_text in zip(vinfo_to_pred, output_texts):
        mid = item["mid"]
        vinfo_dict[mid]["hair_color"] = post_process(output_text)

    with open(output_path, "w") as f:
        json.dump(vinfo, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    from utils import CLI

    CLI(run)
