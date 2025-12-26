import json
import os
import torch
import yaml
from diffusers import DiffusionPipeline
from tqdm import tqdm

# --- 配置文件加载 ---
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_CONFIG_PATH = os.path.join(CURRENT_SCRIPT_DIR, 'config.yaml')

def load_paths():
    """从配置加载输入 JSON 路径"""
    if not os.path.exists(LOCAL_CONFIG_PATH):
        raise FileNotFoundError(f"配置文件未找到: {LOCAL_CONFIG_PATH}")
    with open(LOCAL_CONFIG_PATH, 'r', encoding='utf-8') as f:
        conf = yaml.safe_load(f)
    
    json_input = conf['paths']['output_json']
    if not os.path.isabs(json_input):
        json_input = os.path.abspath(os.path.join(CURRENT_SCRIPT_DIR, json_input))
    return json_input, conf

def main():
    json_path, conf = load_paths()
    
    # 1. 初始化 SDXL 1.0 模型
    print("正在加载 SDXL 1.0 模型...")
    pipe = DiffusionPipeline.from_pretrained(
        conf['SD'], 
        torch_dtype=torch.float16, 
        use_safetensors=True, 
        variant="fp16"
    )
    pipe.to("cuda")

    # 2. 启用 torch.compile 加速 (要求 torch >= 2.0)
    print("正在应用 torch.compile 优化...")
    pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

    # 3. 加载数据集
    with open(json_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    # 4. 统一输出目录设置
    # 修改点：所有背景图统一存放在 data/imgs/SD/
    sd_img_root = os.path.join(CURRENT_SCRIPT_DIR, "imgs", "SD")
    os.makedirs(sd_img_root, exist_ok=True)

    print(f"🚀 开始生成背景底图，目标目录: {sd_img_root}")

    for item in tqdm(data_list, desc="Generating"):
        if item.get("status") != "success":
            continue

        img_id = item["id"]  # 这里的 ID 已在 01 脚本中设为从 1 开始
        key_phrase = item["key_phrase"] # [cite: 25, 124]

        # 最终保存路径：data/imgs/SD/{id}.jpg
        save_path = os.path.join(sd_img_root, f"{img_id}.jpg")

        if os.path.exists(save_path):
            continue

        # 提示词模板
        prompt = f"A photo of {key_phrase}"

        with torch.no_grad():
            # 生成 1024x1024 图像，匹配后续脚本需求 [cite: 512, 574]
            image = pipe(prompt=prompt).images[0]
        
        image.save(save_path)

    print(f"\n✅ 图像生成完成！共保存 {len(data_list)} 张图片。")

if __name__ == "__main__":
    main()