import json
import os
from typing import Any


def load_config(config_path="configs.json") -> dict:
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

def dump_config(config: dict, config_path="configs.json"):
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def load_vinfo(path: str = "data/vtuber_info_20241204.json") -> list:
    with open(path, "r") as f:
        vinfo = json.load(f)

    return vinfo


def get_file_type(url: str) -> str:
    file_name = os.path.basename(url)
    return file_name.split(".")[-1]


def prefilter(vinfo: list) -> list:
    config = load_config()

    follower_threshold = config["follower_threshold"]
    guard_threshold = config["guard_threshold"]

    accept_file_type = config["accept_file_type"]

    filtered_vinfo = []
    for item in vinfo:
        if (
            item["follower"] >= follower_threshold
            and item["guardNum"] >= guard_threshold
            and get_file_type(item["face"]) in accept_file_type
        ):
            filtered_vinfo.append(item)

    return filtered_vinfo


def CLI(*args: Any, **kwargs: Any) -> Any:
    from jsonargparse import CLI

    kwargs.setdefault("as_positional", False)

    return CLI(*args, **kwargs)


def load_model_and_processor(path: str) -> tuple:
    from transformers import (
        Qwen2VLForConditionalGeneration,
        AutoTokenizer,
        AutoProcessor,
    )

    # default: Load the model on the available device(s)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        path, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(path)

    return model, processor
