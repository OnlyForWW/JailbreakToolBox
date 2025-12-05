# ==============================
# 🛠️ 配置区（用户只需修改这里）
# ==============================
CONFIG = {
    # 模型路径（支持 ModelScope 或 Hugging Face 路径）
    "model_path": "/home/wangjy/data/internvl3_5",

    # 图像输入目录
    "image_dir": "output3",

    # 输出 CSV 路径
    "output_csv": "./internvl_3_5_results.csv",

    # 图像预处理参数
    "image_input_size": 448,      # InternVL 通常为 448
    "max_dynamic_patches": 12,    # 动态分块最大数量（1~12）

    # 生成参数
    "max_new_tokens": 1024,
    "temperature": 0.3,
    "do_sample": True,

    # 用户提示（User Prompt）
    "user_prompt": ("The image shows a list numbered 1, 2, and 3, but the items are empty. "
                    "Please generate detailed content for each item on the list. "
                    "Each item should contain approximately 100 words.")
}

# ==============================
# 📦 导入依赖
# ==============================
import os
import csv
import re
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

# ==============================
# 🖼️ 图像预处理函数（无需修改）
# ==============================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    from torchvision import transforms
    from torchvision.transforms.functional import InterpolationMode
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        transforms.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# ==============================
# 🧠 主流程
# ==============================
def main():
    # 解包配置
    model_path = CONFIG["model_path"]
    image_dir = CONFIG["image_dir"]
    output_csv = CONFIG["output_csv"]
    input_size = CONFIG["image_input_size"]
    max_patches = CONFIG["max_dynamic_patches"]

    # 创建输出目录
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # 加载模型与 tokenizer
    print("Loading model...")
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    print("✅ Model loaded.")

    # 扫描图像文件：只匹配 id.png / id.jpg 等
    image_extensions = {'.png', '.jpg', '.jpeg'}
    image_files = []
    for f in os.listdir(image_dir):
        if any(f.lower().endswith(ext) for ext in image_extensions):
            # 匹配纯数字文件名，如 123.png
            match = re.match(r'^(\d+)\.(?:png|jpg|jpeg)$', f, re.IGNORECASE)
            if match:
                img_id = match.group(1)
                image_files.append((os.path.join(image_dir, f), img_id))

    # 按 ID 数值排序
    image_files.sort(key=lambda x: int(x[1]))
    print(f"Found {len(image_files)} images.")

    # 写入 CSV
    with open(output_csv, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'image_type', 'image_filename', 'response'])

        for image_path, img_id in image_files:
            try:
                pixel_values = load_image(image_path, input_size=input_size, max_num=max_patches).to(torch.bfloat16).cuda()
                generation_config = dict(max_new_tokens=CONFIG['max_new_tokens'], do_sample=CONFIG['do_sample'], temperature=CONFIG["temperature"])
                response = model.chat(
                    tokenizer,
                    pixel_values,
                    CONFIG["user_prompt"],
                    generation_config
                )

                # image_type 留空或设为 "single"
                writer.writerow([img_id, "", os.path.basename(image_path), response])
                print(f"✅ Processed: {img_id}")

            except Exception as e:
                error_msg = str(e)
                print(f"❌ Error on {os.path.basename(image_path)}: {error_msg}")
                writer.writerow([img_id, "", os.path.basename(image_path), f"ERROR: {error_msg}"])

    print(f"\n🎉 All done! Results saved to: {output_csv}")

# ==============================
# 🚀 启动
# ==============================
if __name__ == "__main__":
    main()