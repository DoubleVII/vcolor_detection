from tqdm import tqdm
from qwen_vl_utils import process_vision_info


def _batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


def run(messages_list: list, batch_size: int, model, processor):

    output_texts = []
    with tqdm(total=len(messages_list)) as pbar:
        for batch_messages in _batch(messages_list, batch_size):

            texts = [
                processor.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=True
                )
                for msg in batch_messages
            ]
            image_inputs, video_inputs = process_vision_info(batch_messages)
            inputs = processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            # Batch Inference
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            batch_output_texts = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            output_texts.extend(batch_output_texts)
            pbar.update(len(batch_messages))
    return output_texts
