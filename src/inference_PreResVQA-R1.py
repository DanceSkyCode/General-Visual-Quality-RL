import os
import random
import re
from tqdm import tqdm
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
file_name = ""
file_name_think = ""
MODEL_PATH = ""
video_root = ""
label_file = ""

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=device,
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
processor.tokenizer.padding_side = "left"
def preprocess_image_keep_aspect(image_path, max_size=500):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    scale = max_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    return img
def get_image_groups(label_file, video_root):
    image_groups = []
    missing_videos = []
    with open(label_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            video_rel, _ = line.split(",")
            base_path = os.path.join(video_root, os.path.splitext(video_rel.strip())[0])
            image_paths = [f"{base_path}_{i}.png" for i in range(4)]
            if all(os.path.exists(p) for p in image_paths):
                image_groups.append(image_paths)
            else:
                missing_videos.append(video_rel)
    if missing_videos:
        print(f"⚠️ Missing images for {len(missing_videos)} videos, skipped:")
        for v in missing_videos:
            print("  -", v)
    return image_groups
def score_batch_image(image_groups, model, processor, batch_size=4):
    question_prompt = (
        "You are doing the video quality assessment task. Please compare the multi-frame images and evaluate their global temporal smoothness. "
        "Then compare and describe the rest magnifying local single frame pictures in five parts: saturation rating; granularity rating; sharpness rating; foreground rating; and background rating. "
        "Then, please rate both global and local quality of this video. "
        "The ratings should be two float scores between 1 and 5, rounded to two decimal places, with 1 representing very poor quality and 5 representing excellent quality."
    )
    QUESTION_TEMPLATE = "{Question} First output the thinking process in <think> </think> tags and then output the final answer with only one score in <answer> </answer> tags."
    path_score_dict = {}
    open(file_name_think, "w", encoding="utf-8").close()
    open(file_name, "w", encoding="utf-8").close()
    for i in tqdm(range(0, len(image_groups), batch_size), desc="Processing batches"):
        batch_groups = image_groups[i:i+batch_size]
        messages = []

        # 构造 batch 的 prompt
        for image_paths in batch_groups:
            content_items = []
            for idx, path in enumerate(image_paths):
                if idx == 0:
                    content_items.extend([
                        {'type': 'text', 'text': 'Global multiframe picture:'},
                        {'type': 'image', 'image': path}
                    ])
                else:
                    content_items.extend([
                        {'type': 'text', 'text': f'Local single frame picture {idx}:'},
                        {'type': 'image', 'image': path}
                    ])
            content_items.append({'type': 'text', 'text': QUESTION_TEMPLATE.format(Question=question_prompt)})
            messages.append({"role": "user", "content": content_items})
        text_list = [processor.apply_chat_template([msg], tokenize=False, add_generation_prompt=True, add_vision_id=True) for msg in messages]
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=text_list, images=image_inputs, padding=True, return_tensors="pt").to(device)
        generated_ids = model.generate(
            **inputs,
            use_cache=True,
            max_new_tokens=512,
            do_sample=True,
            top_k=50,
            top_p=1
        )
        for j, out_ids in enumerate(generated_ids):
            out_trimmed = out_ids[len(inputs.input_ids[j]):]
            output_text = processor.batch_decode([out_trimmed], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            reasoning = re.findall(r'<think>(.*?)</think>', output_text, re.DOTALL)
            reasoning = reasoning[-1].strip() if reasoning else ""
            try:
                model_output_matches = re.findall(r'<answer>(.*?)</answer>', output_text, re.DOTALL)
                if model_output_matches:
                    numbers = [float(x) for x in re.findall(r'\d+\.?\d*', model_output_matches[-1])]
                    score = sum(numbers)/len(numbers) if numbers else random.randint(1, 5)
                else:
                    score = random.randint(1, 5)
            except:
                score = random.randint(1, 5)
            path = batch_groups[j][0]
            path_score_dict[path] = score
            with open(file_name_think, "a", encoding="utf-8") as f:
                f.write(f"[IMAGES]: {batch_groups[j]}\n")
                f.write(f"<think>\n{reasoning}\n</think>\n")
                f.write(f"<answer>\n{model_output_matches[-1].strip() if model_output_matches else ''}\n</answer>\n")
                f.write("="*80 + "\n\n")
            with open(file_name, "a", encoding="utf-8") as f:
                f.write(f"{path} {score}\n")
    return path_score_dict
if __name__ == "__main__":
    image_groups = get_image_groups(label_file, video_root)
    print(f"Total video samples: {len(image_groups)}")
    path_score_dict = score_batch_image(image_groups, model, processor, batch_size=16)
    print("✅ Done! Results saved:")
    print(f"  - {file_name}")
    print(f"  - {file_name_think}")
