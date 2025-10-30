import os
import re
import pathlib
import random
import pandas as pd
import numpy as np
import itertools
from scipy.stats import spearmanr
from datetime import datetime
from transformers import AutoModel
from transformers import AutoConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from dataclasses import dataclass, field
from typing import Optional
from babel.numbers import parse_decimal
from utils.math import compute_score
from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration
import pathlib
from accelerate import Accelerator
from transformers import AutoModelForCausalLM
from math_verify import parse, verify
from open_r1.trainer.grpo_trainer_secondstage import VLMGRPOTrainer, GRPOConfig
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
import PIL
from itertools import combinations
from Levenshtein import ratio
from open_r1.utils.pycocotools.coco import COCO
from open_r1.utils.pycocotools.cocoeval import COCOeval
import json
import math
from json_repair import repair_json
from peft import PeftConfig, get_peft_model
from open_r1.vlm_modules import *
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionFlashAttention2, apply_rotary_pos_emb_flashatt, flash_attn_varlen_func
import torch
from typing import Tuple
from transformers.utils import logging
from transformers import AutoProcessor, AutoTokenizer
from openai import OpenAI
logger = logging.get_logger(__name__)

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "sk-proj-1234567890"),
    base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
)
from open_r1.qwen2_5vl_monkey_patch import monkey_patch_qwen2_5vl_flash_attn, monkey_patch_qwen2_5vl_forward
monkey_patch_qwen2_5vl_flash_attn()    
tokenizer = None
def initialize_tokenizer(model_path):
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.
    """
    data_file_paths: str = field(
        default=None,
        metadata={"help": "Paths to data files, separated by ':'"},
    )
    image_folders: str = field(
        default=None,
        metadata={"help": "Paths to image folders, separated by ':'"},
    )
    arrow_cache_dir: str = field(
        default=None,
        metadata={"help": "Path to arrow cache directory"},
    )
    val_split_ratio: float = field(
        default=0.0,
        metadata={"help": "Ratio of validation split, default 0.0"},
    )
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image (for QwenVL)"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image (for QwenVL)"},
    )
    max_anyres_num: Optional[int] = field(
        default=12,
        metadata={"help": "Maximum number of anyres blocks for the image (for InternVL)"},
    )
    reward_method: Optional[str] = field(
        default=None,
        metadata={
            "help": "Choose reward method: 'default', 'mcp', ..."
        },
    )
    question_template: Optional[str] = field(
        default="scoring",
        metadata={
            "help": "Choose scoring or comparing question"
        },
    )
def extract_first_number(model_answer):
    match = re.search(r'-?\d+(\.\d+)?', model_answer)
    if match:
        return float(match.group())
    else:
        return random.randint(1, 5)
def fidelity_reward(pred1, pred2, var1, var2, gt, device):
    esp = 1e-6
    try:
        normal_dist = torch.distributions.Normal(0, 1)
        _cur = (pred1 - pred2) / torch.sqrt(var1 + var2 + esp)
        p = normal_dist.cdf(_cur)
    except:
        print("Meet Error ...")
        p = torch.tensor(0.5, dtype=torch.float32, device=device)
    
    reward = torch.sqrt(p * gt + esp) + torch.sqrt((1 - p) * (1 - gt) + esp)
    return reward
def accuracy_reward(completions, solution, **kwargs):
    device = kwargs.get("device", None)
    n_gen = kwargs.get("num_generations", 6)
    min_std = float(kwargs.get("min_std", 0.5))
    lambda_std = float(kwargs.get("lambda_std", 0.5))   
    flat_solutions = solution  
    reshaped_solution = []     

    for i in range(0, len(flat_solutions), n_gen):
        block = flat_solutions[i:i + n_gen]  
        per_example = []
        for s in block:
            try:
                nums = re.findall(r"[-+]?\d*\.\d+|\d+", str(s))
                if len(nums) >= 1:
                    val = float(nums[0])
                else:
                    val = 3.0
            except Exception:
                val = 3.0
            per_example.append(val)
        reshaped_solution.append(per_example)  
    contents = [completion[0]["content"] for completion in completions]
    reshaped_content = [contents[i:i + n_gen] for i in range(0, len(contents), n_gen)]
    batch_pred = []    
    batch_mean = []    
    batch_var = []       
    batch_std_penalty = []
    batch_triple_compare = []
    for i in range(len(reshaped_content)):
        cur_pred_list = []
        cur_std_penalties = []
        cur_intra_triple_compare = []
        for j in range(len(reshaped_content[i])):
            text = reshaped_content[i][j]
            try:
                nums = re.findall(r"[-+]?\d*\.\d+|\d+", str(text))
                nums = [float(x) for x in nums]
                if len(nums) >= 2:
                    last5 = nums[-2:]
                elif len(nums) > 0:
                    last5 = nums + [nums[0]] * (2 - len(nums))
                else:
                    last5 = [3.0] * 2
                cur_intra_triple_compare.append(last5)
                last5_tensor = torch.tensor(last5, dtype=torch.float32, device=device)
                pred_mean = float(last5_tensor.mean().item())
                dim_std = torch.std(last5_tensor, unbiased=False)
                if dim_std < min_std:
                    std_penalty = lambda_std * (min_std - dim_std) #lambda
                else:
                    std_penalty = 0.0
            except Exception:
                pred_mean = 3.0
                std_penalty = 0.0
            cur_pred_list.append(pred_mean)
            cur_std_penalties.append(float(std_penalty))
        batch_std_penalty.append(cur_std_penalties)
        batch_pred.append(cur_pred_list)
        cur_pred_tensor = torch.tensor(cur_pred_list, dtype=torch.float32, device=device)
        if cur_pred_tensor.numel() == 0:
            p_mean = torch.tensor(3.0, dtype=torch.float32, device=device)
            p_var = torch.tensor(0.0, dtype=torch.float32, device=device)
        else:
            p_mean = torch.mean(cur_pred_tensor)
            if cur_pred_tensor.size(0) > 1:
                p_var = torch.var(cur_pred_tensor, dim=0, unbiased=False)
            else:
                p_var = torch.tensor(0.0, dtype=torch.float32, device=device)
        batch_mean.append(p_mean)
        batch_var.append(p_var)


        J = len(cur_intra_triple_compare)
        if J < 3:
            return torch.zeros(J, dtype=torch.float32, device=device)
        sample_triples = list(combinations(range(J), 3))  
        num_triples = len(sample_triples) 
        num_element_dims = 2  
        gain = torch.zeros(J, dtype=torch.int32, device=device)
        sample_triple_count = torch.zeros(J, dtype=torch.int32, device=device)
        for triple in sample_triples:
            j1, j2, j3 = triple
            sample_triple_count[[j1, j2, j3]] += 1
            for elem_dim in range(num_element_dims):
                val1 = cur_intra_triple_compare[j1][elem_dim]
                val2 = cur_intra_triple_compare[j2][elem_dim]
                val3 = cur_intra_triple_compare[j3][elem_dim]
                val_sample_pairs = [(val1, j1), (val2, j2), (val3, j3)]
                sorted_pairs = sorted(val_sample_pairs, key=lambda x: x[0])
                centroid_val, centroid_sample = sorted_pairs[1]
                gain[centroid_sample] += 1
                left_val, left_sample = sorted_pairs[0]
                right_val, right_sample = sorted_pairs[2]
                if abs(left_val - centroid_val) < 0.05:
                    gain[left_sample] += 1
                if abs(right_val - centroid_val) < 0.05:
                    gain[right_sample] += 1
        max_possible_gain = sample_triple_count * num_element_dims  # [J,]
        reward = torch.where(
            max_possible_gain > 0,  
            gain.float() / max_possible_gain.float(),  
            torch.tensor(0.0, device=device) 
        )
        lambda_triple = float(kwargs.get("lambda_triple", 0.5)) 
        reward = lambda_triple * reward
        batch_triple_compare.append(reward) # without lambda
    batch_size = len(batch_pred)
    rewards = []
    sorted_pred_with_indices = []
    for i in range(batch_size):
        pred_with_idx = [(val, idx) for idx, val in enumerate(batch_pred[i])]
        sorted_pred = sorted(pred_with_idx, key=lambda x: x[0])
        sorted_pred_with_indices.append(sorted_pred)
    for i in range(batch_size):
        K = len(sorted_pred_with_indices[i])
        sorted_rewards = [0.0] * K
        for j in range(K):
            pred1_val, _ = sorted_pred_with_indices[i][j]
            if batch_size <= 1:
                sorted_rewards[j] = 0.0
                continue
            reward_sum = 0.0
            count = 0
            pred1_val = torch.tensor(pred1_val, dtype=torch.float32, device=device)
            for z in range(batch_size):
                if z == i:
                    continue
                pred2_val, _ = sorted_pred_with_indices[z][j]
                pred2_val = torch.tensor(pred2_val, dtype=torch.float32, device=device)
                gt_i = float(reshaped_solution[i][0])
                gt_z = float(reshaped_solution[z][0])
                if gt_i > gt_z:
                    gt_rel = 1
                elif gt_i < gt_z:
                    gt_rel = 0
                else:
                    gt_rel = 0.5
                if pred1_val > pred2_val:
                    pred_rel = 1
                elif pred1_val < pred2_val:
                    pred_rel = 0
                else:
                    pred_rel = 0.5
                compare_metric = 1.0 if (pred_rel == gt_rel) else 0
                gt_abs_diff = abs(gt_i - gt_z)
                denom = gt_abs_diff + abs(pred1_val - gt_i) + abs(pred2_val - gt_z)
                score_metric = gt_abs_diff / denom
                # _r_val = compare_metric * score_metric + (1 - compare_metric) * (1 - score_metric)
                _r_val = torch.sqrt(compare_metric * score_metric) + torch.sqrt((1 - compare_metric) * (1 - score_metric))
                # _r_val = torch.sqrt(compare_metric * torch.exp(score_metric)) + torch.sqrt((1 - compare_metric) * torch.exp(( -1 - score_metric)))
                reward_sum += _r_val
                count += 1
            final_reward_wopenality = reward_sum / max(1, count)
            sorted_rewards[j] = final_reward_wopenality
        original_order_rewards = [0.0] * K
        for sorted_j in range(K):
            _, original_j = sorted_pred_with_indices[i][sorted_j]
            original_order_rewards[original_j] = sorted_rewards[sorted_j]
        reward1 = torch.tensor(original_order_rewards, dtype=torch.float32, device=device) 
        reward2 = torch.tensor(batch_std_penalty[i], dtype=torch.float32, device=device) 
        reward3 = torch.tensor(batch_triple_compare[i], dtype=torch.float32, device=device) 
        batch_size = len(batch_pred)
        K = len(batch_pred[0])
        sorted_triple_rewards = [ [ [] for _ in range(K) ] for _ in range(batch_size) ]
        for j in range(K): 
            anchor_pred = float(batch_pred[i][j])
            anchor_gt   = float(reshaped_solution[i][0])
            other_indices = [idx for idx in range(batch_size) if idx != i]
            all_pairs = list(itertools.combinations(other_indices, 2))
            for a, b in all_pairs:
                pred_a = float(batch_pred[a][j])
                pred_b = float(batch_pred[b][j])
                gt_a   = float(reshaped_solution[a][0])
                gt_b   = float(reshaped_solution[b][0])
                triplet = [(anchor_gt, anchor_pred), (gt_a, pred_a), (gt_b, pred_b)]
                pred_vals = [triplet[i][1] for i in range(3)]
                gt_vals   = [triplet[i][0] for i in range(3)]
                pair_indices = [(0,1), (1,2), (0,2)]
                reward_sum = 0
                for mm,nn in pair_indices:
                    pred_rel = pred_vals[mm] > pred_vals[nn]
                    gt_rel   = gt_vals[mm] > gt_vals[nn]
                    reward_sum += 1.0 if pred_rel == gt_rel else 0.0
                reward = reward_sum / 3.0
                sorted_triple_rewards[i][j].append(reward)
        final_triplet_rewards = [ torch.tensor(sorted_triple_rewards[i][j], dtype=torch.float32, device=device).mean().item()
                                    for j in range(K) ] 
        final_triplet_tensor = torch.tensor(final_triplet_rewards, dtype=torch.float32, device=device)
        reward =  reward1 * 0.375 - reward2 + 0.5 * reward3 + 0.125 * final_triplet_tensor
        rewards.extend(reward.tolist())
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            image_path = kwargs.get("image_path")
            problem = None
            try:
                problem = kwargs.get("problem")[0]
            except Exception:
                problem = None
            try:
                img_block = [image_path[k:k + n_gen] for k in range(0, len(image_path), n_gen)] if image_path else None
            except Exception:
                img_block = None
            with open(log_path, "a", encoding='utf-8') as f:
                for j in range(K):
                    f.write(f"------------- {current_time} Std_penality: {batch_std_penalty[i][j]} -------------\n")
                    f.write(f"------------- {current_time} triple reward: {batch_triple_compare[i][j]} -------------\n")
                    f.write(f"------------- {current_time} Accuracy reward: {reward[j]} -------------\n")
                    if img_block:
                        try:
                            f.write(f"image_path: {img_block[i][j]}\n")
                        except Exception:
                            f.write(f"image_path: {img_block}\n")
                    f.write(f"problem: {problem}\n")
                    f.write(f"Content: {reshaped_content[i][j]}\n")
                    try:
                        f.write(f"Solution_scalar: {reshaped_solution[i][j] if j < len(reshaped_solution[i]) else reshaped_solution[i][0]}\n")
                    except Exception:
                        pass
    return rewards

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>(?:\s*\d+\.?\d*\s*\n){2}\s*</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]

    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    if os.getenv("DEBUG_MODE") == "true":
        log_path = os.getenv("LOG_PATH")
        with open(log_path.replace(".txt", "_format.txt"), "a", encoding='utf-8') as f:
            f.write(f"------------- {current_time} Format reward -------------\n")
            for content, match in zip(completion_contents, matches):
                f.write(f"Content: {content}\n")
                f.write(f"Has format: {bool(match)}\n")

    return [1.0 if match else 0.0 for match in matches]
reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward
}
@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False
def get_vlm_module(model_name_or_path):
    if "qwen" in model_name_or_path.lower():
        return Qwen2VLModule
    elif "internvl" in model_name_or_path.lower():
        return InvernVLModule
    else:
        raise ValueError(f"Unsupported model: {model_name_or_path}")
def main(script_args, training_args, model_args):
    # Load the VLM module
    vlm_module_cls = get_vlm_module(model_args.model_name_or_path)
    print("using vlm module:", vlm_module_cls.__name__)
    question_prompt = vlm_module_cls.get_question_template(task_type=script_args.question_template) # scoring

    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs] # ["accuracy", "format"]
    print("reward_funcs:", reward_funcs)

    # Load the JSONL datasets
    import json
    from datasets import Dataset
    
    data_files = script_args.data_file_paths.split(":")
    image_folders = script_args.image_folders.split(":")

    training_args.max_completion_length = 256
    
    if len(data_files) != len(image_folders):
        raise ValueError("Number of data files must match number of image folders")
    
    if script_args.reward_method is None:
        accu_reward_methods = ["default"] * len(data_files)
    else:
        accu_reward_methods = script_args.reward_method.split(":")
        assert len(accu_reward_methods) == len(data_files), f"Number of reward methods must match number of data files: {len(accu_reward_methods)} != {len(data_files)}"
    
    
    if len(data_files) != len(image_folders):
        raise ValueError("Number of data files must match number of image folders")
    all_data = []
    prompts = [
            "You are doing the video quality assessment task. Please compare the internal packed video frames in the first picture and describe the global multiframe pictures in temporal deminsion and global scene. "
            "Then compare and describe the rest local single frame pictures in five parts: saturation rating; granularity rating; sharpness rating; foreground rating; and background rating. "
            "Finally, please rate two scores in the global multiframe and local single frame dimension. "
            "All ratings should be floats between 1 and 5, where 1 represents very poor quality and 5 represents excellent quality."
    ]
    for data_file, image_folder, accu_reward_method in zip(data_files, image_folders, accu_reward_methods):
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)  
            for item in data:
                if 'image' in item:
                    image_entries = item['image'][:4]
                    if isinstance(image_entries, str):
                        image_entries = [image_entries]
                    item['image_path'] = image_entries
                item['problem'] = item['conversations'][0]['value'].replace('<image>', '')
                solution_value = item['conversations'][1]['value']
                if isinstance(solution_value, str):
                    item['solution'] = solution_value.replace('<answer>', '').replace('</answer>', '').strip()
                else:
                    item['solution'] = str(solution_value)

                del item['conversations']
                item['accu_reward_method'] = item.get('accu_reward_method', accu_reward_method)
                all_data.append(item)
    dataset = Dataset.from_list(all_data)
    def make_conversation_from_jsonl(example):
        try:
            image_paths = example.get('image_path', [])
            if image_paths is None or not isinstance(image_paths, list):
                print(f"[WARN] id={example.get('id')} image_path invalid type: {type(image_paths)}")
                image_paths = [image_paths] if image_paths else []
            content_items = []
            for idx, path in enumerate(image_paths):
                if idx == 0:
                    content_items.extend([
                        {'type': 'text', 'text': 'Global multiframe picture:'},
                        {'type': 'image', 'text': path}
                    ])
                else:
                    content_items.extend([
                        {'type': 'text', 'text': f'Local single frame picture {idx}:'},
                        {'type': 'image', 'text': path}
                    ])
            content_items.append({
                'type': 'text',
                'text': question_prompt.format(Question=example.get('problem', ''))
            })
            return {
                'image_path': image_paths,
                'dataset_name': example.get('dataset_name', 'Unknown'),
                'problem': example.get('problem', ''),
                'solution': f"<answer> {example.get('solution', '')} </answer>",
                'accu_reward_method': example.get('accu_reward_method', ''),
                'prompt': [{'role': 'user', 'content': content_items}]
            }
        except Exception as e:
            print(f"[ERROR] id={example.get('id')} failed with error: {repr(e)}")
            print(f"Example content: {example}")
            raise e
    dataset = dataset.map(make_conversation_from_jsonl, num_proc=8)
    splits = {'train': dataset}
    if script_args.val_split_ratio > 0:
        train_val_split = dataset.train_test_split(
            test_size=script_args.val_split_ratio
        )
        splits['train'] = train_val_split['train']
        splits['validation'] = train_val_split['test']
    checkpoint_dirs = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
    if checkpoint_dirs:
        latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.name.split("-")[-1]))
        print(f"Resuming from latest checkpoint: {latest_checkpoint}")
    else:
        print("No checkpoint found, using original model path...")
        model_path_for_loading = model_args.model_name_or_path
    config = AutoConfig.from_pretrained(model_path_for_loading)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path_for_loading,
        config=config,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=model_args.attn_implementation
    ).to('cuda')
    processor = AutoProcessor.from_pretrained(model_path_for_loading)
    processor.tokenizer.padding_side = "left"
    print("Initializing tokenizer...")
    initialize_tokenizer(model_args.model_name_or_path)
    trainer_cls = VLMGRPOTrainer
    print("Using trainer:", trainer_cls.__name__)
    trainer = trainer_cls(
        model=model,
        model_name = model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        vlm_module=vlm_module_cls(),
        train_dataset=splits['train'],
        eval_dataset=splits.get('validation') if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        processing_class=processor,  
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub()
if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    if training_args.deepspeed and "zero3" in training_args.deepspeed:
        print("zero3 is used, qwen2_5vl forward monkey patch is applied")
        monkey_patch_qwen2_5vl_forward()
    main(script_args, training_args, model_args)
