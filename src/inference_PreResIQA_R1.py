from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import torch
import random
import re
import os


def get_image_paths(folder_path):
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    image_paths = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext.lower() in image_extensions:
                image_paths.append(os.path.join(root, file))

    return image_paths

def score_batch_image(image_paths, model, processor):
    PROMPT = (
        "You are doing the image quality assessment task. Compare the two distorted images and answer in five parts:"
        "saturation rating; granularity rating; sharpness rating; foreground rating; and background rating." 
        "All ratings should be floats between 1 and 5, rounded to two decimals, where 1 represents very poor quality and 5 represents excellent quality."
    )

    QUESTION_TEMPLATE = "{Question} First output the thinking process in <think> </think> tags and then output the final answer with only one score in <answer> </answer> tags."

    messages = []
    for img_path in image_paths:
        message = [
            {
                "role": "user",
                "content": [
                    {'type': 'image', 'image': img_path},
                    {"type": "text", "text": QUESTION_TEMPLATE.format(Question=PROMPT)}
                ],
            }
        ]
        messages.append(message)

    BSZ = 32
    all_outputs = []  # List to store all answers
    for i in tqdm(range(0, len(messages), BSZ)):
        batch_messages = messages[i:i + BSZ]
    
        # Preparation for inference
        text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, add_vision_id=True) for msg in batch_messages]
        
        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=512, do_sample=True, top_k=50, top_p=1)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        batch_output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        all_outputs.extend(batch_output_text)
    
    path_score_dict = {}
    for img_path, model_output in zip(image_paths, all_outputs):
        reasoning = re.findall(r'<think>(.*?)</think>', model_output, re.DOTALL)
        reasoning = reasoning[-1].strip()

        try:
            model_output_matches = re.findall(r'<answer>(.*?)</answer>', model_output, re.DOTALL)
            model_answer = model_output_matches[-1].strip() if model_output_matches else model_output.strip()
            score = float(re.search(r'\d+(\.\d+)?', model_answer).group())
        except:
            print(f"Meet error with {img_path}, please generate again.")
            score = random.randint(1, 5)

        path_score_dict[img_path] = score

    return path_score_dict


random.seed(1)
MODEL_PATH = "your_PreResIQA_R1_path"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=device,
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
processor.tokenizer.padding_side = "left"

image_root = "test_images"
image_paths = get_image_paths(image_root) # It should be a list

path_score_dict = score_batch_image(
    image_paths, model, processor
)

file_name = "output.txt"
with open(file_name, "w") as file:
    for key, value in path_score_dict.items():
        file.write(f"{key} {value}\n") 

print("Done!")