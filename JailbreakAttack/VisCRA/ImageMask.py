# generate_masks.py
import os
import sys
import json
import yaml
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageDraw
import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from qwen_vl_utils import process_vision_info

# 全局变量（将从 config.yaml 加载）
CONFIG = None
VISION_TOKEN = None
GRID_DIM = None
GAUSSIAN_SIGMA = None
ATTENTION_BLOCK_SIZE = None
ATTENTION_LAYER = None
OUTPUT_ROOT = None

def load_config(config_path):
    global CONFIG, VISION_TOKEN, GRID_DIM, GAUSSIAN_SIGMA
    global ATTENTION_BLOCK_SIZE, ATTENTION_LAYER, OUTPUT_ROOT
    with open(config_path, 'r', encoding='utf-8') as f:
        CONFIG = yaml.safe_load(f)
    
    viscra = CONFIG["viscra"]
    VISION_TOKEN = viscra["vision_token"]
    GRID_DIM = viscra["grid_dim"]
    GAUSSIAN_SIGMA = viscra["gaussian_sigma"]
    ATTENTION_BLOCK_SIZE = viscra["attention_block_size"]
    ATTENTION_LAYER = viscra["attention_layer"]
    OUTPUT_ROOT = viscra["output_root"]

def find_top3_blocks_with_stride(matrix, block_size=12, stride=3, max_overlap=40):
    integral = np.cumsum(np.cumsum(matrix, axis=0), axis=1)
    integral = np.pad(integral, ((1, 0), (1, 0)), mode='constant')
    rows, cols = matrix.shape
    candidates = []
    for i in range(0, rows - block_size + 1, stride):
        for j in range(0, cols - block_size + 1, stride):
            total = (
                integral[i + block_size, j + block_size]
                - integral[i, j + block_size]
                - integral[i + block_size, j]
                + integral[i, j]
            )
            candidates.append((-total, i, j))
    candidates.sort()
    selected = []
    for cand in candidates:
        _, x, y = cand
        conflict = False
        for (s_i, s_j) in selected:
            x_overlap = max(0, min(x + block_size, s_i + block_size) - max(x, s_i))
            y_overlap = max(0, min(y + block_size, s_j + block_size) - max(y, s_j))
            if x_overlap * y_overlap >= max_overlap:
                conflict = True
                break
        if not conflict:
            selected.append((x, y))
            if len(selected) >= 3:
                break
    return selected[:3]

def visualize_attention(image_input, attention_tensor, block_size, output_dir, color="green"):
    os.makedirs(output_dir, exist_ok=True)
    original_path = os.path.join(output_dir, "original.png")
    image_input.save(original_path)

    img_width, img_height = image_input.size
    grid_w = img_width // GRID_DIM
    grid_h = img_height // GRID_DIM

    attention_2d = attention_tensor.reshape(grid_h, grid_w)

    # 边缘清零
    attention_2d[:4, :] = 0
    attention_2d[-12:, :] = 0
    attention_2d[:, 0] = 0
    attention_2d[:, -1] = 0

    top3_blocks = find_top3_blocks_with_stride(attention_2d, block_size=block_size, stride=3)
    mask_paths = []
    for idx, (block_row, block_col) in enumerate(top3_blocks):
        masked_img = image_input.copy()
        draw = ImageDraw.Draw(masked_img)
        mask_x = block_col * GRID_DIM
        mask_y = block_row * GRID_DIM
        mask_width = block_size * GRID_DIM
        mask_height = block_size * GRID_DIM
        mask_x_end = min(mask_x + mask_width, img_width)
        mask_y_end = min(mask_y + mask_height, img_height)
        draw.rectangle((mask_x, mask_y, mask_x_end, mask_y_end), fill=color)
        mask_path = os.path.join(output_dir, f"mask_block{idx}.png")
        masked_img.save(mask_path)
        mask_paths.append(mask_path)

    # Heatmap
    plt.figure(figsize=(img_width / 100, img_height / 100), dpi=100)
    plt.imshow(attention_2d, cmap="viridis", aspect="auto", extent=[0, img_width, img_height, 0])
    plt.axis("off")
    heatmap_path = os.path.join(output_dir, "heatmap.png")
    plt.savefig(heatmap_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    # Overlay
    upsampled = attention_2d.repeat(GRID_DIM, axis=0).repeat(GRID_DIM, axis=1)
    upsampled = upsampled[:img_height, :img_width]
    smoothed = gaussian_filter(upsampled, sigma=GAUSSIAN_SIGMA)
    normalized = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min() + 1e-8)
    plt.figure(figsize=(img_width / 100, img_height / 100), dpi=100)
    plt.imshow(image_input)
    plt.imshow(normalized, cmap="viridis", alpha=0.5, extent=[0, img_width, img_height, 0])
    plt.axis("off")
    overlay_path = os.path.join(output_dir, "overlay.png")
    plt.savefig(overlay_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    return {
        "original": original_path,
        "heatmap": heatmap_path,
        "overlay": overlay_path,
        "masks": mask_paths,
    }

def load_model(model_path, device):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map=device,
    ).eval()
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, padding_side="left", use_fast=True
    )
    return model, processor

def prepare_inputs(processor, query: str, image: str):
    placeholders = [{"type": "image", "image": image}]
    messages = [
        {
            "role": "user",
            "content": [
                *placeholders,
                {"type": "text", "text": f"{query} Answer:"},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    return prompt, image_inputs

def generate(model, processor, query, image_path):
    text_inputs, image_inputs = prepare_inputs(processor, query, image_path)
    inputs = processor(
        text=[text_inputs],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    image_token_indices = torch.where(inputs['input_ids'][0] == VISION_TOKEN)[0]
    if len(image_token_indices) == 0:
        print(f"Warning: No vision token found for {image_path}")
        return

    image_token_start = image_token_indices[0].item()
    image_token_end = image_token_indices[-1].item()

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        avg_attn = outputs.attentions[ATTENTION_LAYER][0, :, -1, image_token_start:image_token_end + 1].mean(dim=0).float().cpu().numpy()

        img_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = os.path.join(OUTPUT_ROOT, "mm-safebench", img_name)
        visualize_attention(image_inputs[0], avg_attn, ATTENTION_BLOCK_SIZE, output_dir, color="green")

def main(config_path):
    load_config(config_path)
    
    model_cfg = CONFIG["auxiliary_model"]
    dataset_cfg = CONFIG["dataset"]

    # 加载模型
    model, processor = load_model(model_cfg["path"], model_cfg["device"])

    # 加载数据集
    with open(dataset_cfg["input_json"], 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    if not isinstance(dataset, list):
        raise ValueError("Input JSON must be a list of samples.")

    for item in tqdm(dataset, desc="Generating Attention Masks"):
        query = item.get("rephrased_question_sd")
        image_path = item.get("image")

        if not query or not image_path or not os.path.exists(image_path):
            print(f"Skipping invalid item: {item.get('id', 'unknown')}")
            continue

        generate(model, processor, query, image_path)
        torch.cuda.empty_cache()

if __name__ == '__main__':
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--config', type=str, required=True, help="Path to config.yaml")
    # args = parser.parse_args()
    CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    LOCAL_CONFIG_PATH = os.path.join(CURRENT_SCRIPT_DIR, 'config.yaml')
    main(LOCAL_CONFIG_PATH)