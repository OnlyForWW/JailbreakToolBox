import json
import os
import yaml
from tqdm import tqdm

# --- 路径与配置加载 ---
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_CONFIG_PATH = os.path.join(CURRENT_SCRIPT_DIR, 'config.yaml')

def load_config():
    """加载本地配置以获取 JSON 路径"""
    if not os.path.exists(LOCAL_CONFIG_PATH):
        raise FileNotFoundError(f"配置文件未找到: {LOCAL_CONFIG_PATH}")
    with open(LOCAL_CONFIG_PATH, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    # 获取 01 脚本生成的 JSON 绝对路径
    json_path = os.path.abspath(os.path.join(CURRENT_SCRIPT_DIR, config['paths']['output_json']))
    
    # 定义最终拼接图像存储的绝对目录 [cite: 574]
    image_dir_abs = os.path.abspath(os.path.join(CURRENT_SCRIPT_DIR, "imgs", "SD_TYPO"))

    if not os.path.exists(json_path):
        print(f"❌ 错误: 找不到 JSON 数据集文件 {json_path}")
        return

    # 1. 加载现有的 JSON 数据
    with open(json_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    print(f"🚀 正在为 {len(data_list)} 条数据关联图像的绝对路径...")

    # 2. 遍历并更新 'image' 字段
    updated_count = 0
    for item in tqdm(data_list):
        # 仅处理文本提取成功的数据
        if item.get("status") != "success":
            continue

        # 构造文件名，ID 对应 1-based 索引
        img_id = item['id']
        filename = f"{img_id}.jpg"
        
        # 获取图像的完整绝对路径
        full_image_path = os.path.join(image_dir_abs, filename)
        
        # 验证物理文件是否存在 [cite: 222]
        if os.path.exists(full_image_path):
            # 修改点：保存为绝对路径
            item['image'] = full_image_path
            updated_count += 1
        else:
            print(f"⚠️ 警告: 未找到对应的拼接图像: {full_image_path}")

    # 3. 保存更新后的 JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)

    print(f"\n✅ 处理完成！已成功关联 {updated_count} 条数据的绝对路径。")
    print(f"📍 数据集位置: {json_path}")

if __name__ == "__main__":
    main()