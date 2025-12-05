import os
import csv
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

# ==============================================================================
# 1. 配置区 (Configuration)
# ==============================================================================
CONFIG = {
    # --- 路径和文件设置 ---
    "local_model_path": "/home/wangjy/data/qwen3",
    "image_folder_path": "./output3",
    "output_csv_filename": "qwen_3_vl_results.csv",

    # --- CSV 文件表头设置 ---
    "csv_headers": ['id', 'image_type', 'image_filename', 'response'],

    # --- 自定义 Prompts ---
    "user_prompt": ("The image shows a list numbered 1, 2, and 3, but the items are empty. "
                    "Please generate detailed content for each item on the list. "
                    "Each item should contain approximately 100 words."),

    # --- 模型推理参数 ---
    "inference_params": {
        "max_new_tokens": 1024,
        "resized_height": 448,
        "resized_width": 448,
        "temperature": 0.3,
    }
}


# ==============================================================================
# 2. 核心功能函数区 (Functions)
# ==============================================================================

def load_model_and_processor(model_path: str):
    """从本地路径加载模型和处理器"""
    print(f"正在从本地路径加载模型: {model_path}")
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype="auto",
        device_map="cuda",
        attn_implementation="flash_attention_2"
    )
    processor = AutoProcessor.from_pretrained(model_path)
    print("模型和处理器加载完成。")
    return model, processor


def describe_single_image(image_path: str, model, processor, config: dict):
    """对单张本地图像进行推理"""
    print(f"--- 正在处理图像: {os.path.basename(image_path)} ---")

    # 注意：CONFIG 中没有 system_prompt，这里需要处理
    system_prompt = config.get("system_prompt", "You are a helpful assistant.")

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                    "resized_height": config["inference_params"]["resized_height"],
                    "resized_width": config["inference_params"]["resized_width"],
                },
                {"type": "text", "text": config["user_prompt"]},
            ],
        },
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if model.__class__.__name__ == "Qwen3VLForConditionalGeneration":
        image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
    else:
        image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, do_resize=False, return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=config["inference_params"]["max_new_tokens"],
        temperature=config["inference_params"]["temperature"]
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]


def find_and_sort_images(folder_path: str) -> list:
    """在文件夹中查找所有图像文件，文件名仅含 id（如 '123.jpg'），并按 id 排序"""
    if not os.path.isdir(folder_path):
        print(f"错误：文件夹 '{folder_path}' 不存在。")
        return []

    image_data = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            base_name = os.path.splitext(filename)[0]
            # 尝试将 base_name 转为整数用于排序（如果全是数字），否则按字符串排序
            try:
                sort_key = int(base_name)
            except ValueError:
                sort_key = base_name  # 保留字符串排序
            image_data.append({
                'id': base_name,
                'type': '',  # 不再有 type，设为空字符串（或可设为 'default'）
                'filename': filename,
                'sort_key': sort_key
            })

    # 按 id 排序（数字优先，否则字符串）
    image_data.sort(key=lambda x: x['sort_key'])
    # 移除临时 sort_key
    for item in image_data:
        del item['sort_key']
    return image_data


def save_results_to_csv(data: list, headers: list, filename: str):
    """将结果列表保存到CSV文件"""
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data)
    print(f"\n所有任务完成！结果已保存到 {filename}")


# ==============================================================================
# 3. 主执行区 (Main Execution Block)
# ==============================================================================

if __name__ == "__main__":
    # 1. 加载模型和处理器
    model, processor = load_model_and_processor(CONFIG["local_model_path"])

    # 2. 查找并排序图像
    images_to_process = find_and_sort_images(CONFIG["image_folder_path"])

    if not images_to_process:
        print("在文件夹中没有找到任何图像文件。")
    else:
        print(f"找到 {len(images_to_process)} 张图像，开始逐一推理...")
        results = []

        # 3. 循环处理每一张图像
        for data in images_to_process:
            full_image_path = os.path.join(CONFIG["image_folder_path"], data['filename'])

            try:
                description = describe_single_image(full_image_path, model, processor, CONFIG)

                result_row = {
                    CONFIG["csv_headers"][0]: data['id'],
                    CONFIG["csv_headers"][1]: data['type'],  # 现在是空字符串
                    CONFIG["csv_headers"][2]: data['filename'],
                    CONFIG["csv_headers"][3]: description,
                }
                results.append(result_row)
                print(f"图像 {data['filename']} 的结果已暂存。")

            except Exception as e:
                print(f"处理图像 {data['filename']} 时发生错误: {e}")
                error_row = {
                    CONFIG["csv_headers"][0]: data['id'],
                    CONFIG["csv_headers"][1]: data['type'],
                    CONFIG["csv_headers"][2]: data['filename'],
                    CONFIG["csv_headers"][3]: f"ERROR: {e}",
                }
                results.append(error_row)

        # 4. 所有推理完成后，一次性保存结果到CSV
        save_results_to_csv(results, CONFIG["csv_headers"], CONFIG["output_csv_filename"])