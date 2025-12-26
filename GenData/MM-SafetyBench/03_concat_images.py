from PIL import Image, ImageFont, ImageDraw
import json
import os
import time
import datetime
import yaml
from tqdm import tqdm

# --- 配置加载 ---
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_CONFIG_PATH = os.path.join(CURRENT_SCRIPT_DIR, 'config.yaml')

def load_config():
    with open(LOCAL_CONFIG_PATH, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

config = load_config()
JSON_PATH = os.path.abspath(os.path.join(CURRENT_SCRIPT_DIR, config['paths']['output_json']))

# 统一目录定义
SD_ROOT = os.path.join(CURRENT_SCRIPT_DIR, "imgs", "SD")
TYPO_ROOT = os.path.join(CURRENT_SCRIPT_DIR, "imgs", "TYPO")
RESULT_ROOT = os.path.join(CURRENT_SCRIPT_DIR, "imgs", "SD_TYPO")

# 字体配置
FONT_SIZE = 90
# 请确保该路径下有字体文件
FONT_PATH = os.path.join(CURRENT_SCRIPT_DIR, 'ARIAL.TTF')
font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
MAX_WIDTH = 1024

def typo_format_text(text):
    """处理文本折行逻辑"""
    img = Image.new('RGB', (MAX_WIDTH, 100), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    word_list = str(text).split(" ")
    word_num = len(word_list)
    formated_text = word_list[0]
    cur_line_len = draw.textlength(formated_text, font=font)
    line_num = 1
    for i in range(1, word_num):
        cur_line_len += draw.textlength(" "+word_list[i], font=font)
        if cur_line_len < MAX_WIDTH:
            formated_text += " "+word_list[i]
        else:
            formated_text += "\n "+word_list[i]
            cur_line_len = draw.textlength(" "+word_list[i], font=font)
            line_num += 1
    return formated_text, line_num

def typo_draw_img(formated_text, line_num, img_path):
    """绘制纯文字图"""
    max_height = FONT_SIZE * (line_num + 1)
    img = Image.new('RGB', (MAX_WIDTH, max_height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, FONT_SIZE/2.0), formated_text, (0, 0, 0), font=font)
    img.save(img_path)

def vertical_concat(image1_path, image2_path, output_path):
    """垂直拼接场景图与文字图"""
    img1 = Image.open(image1_path)
    img2 = Image.open(image2_path)
    w1, h1 = img1.size
    w2, h2 = img2.size

    result_width = max(w1, w2)
    result_height = h1 + h2
    result = Image.new('RGB', (result_width, result_height), (255, 255, 255))

    result.paste(img1, (0, 0))
    result.paste(img2, (0, h1))
    result.save(output_path)

def process_all():
    start_time = time.time()
    
    # 确保目录存在
    os.makedirs(TYPO_ROOT, exist_ok=True)
    os.makedirs(RESULT_ROOT, exist_ok=True)

    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    print(f"开始拼接图像，共 {len(data_list)} 条...")

    for item in tqdm(data_list):
        if item.get("status") != "success":
            continue

        img_id = item['id']
        key_phrase = item['key_phrase']

        # 1. 生成文字图
        typo_path = os.path.join(TYPO_ROOT, f"{img_id}.jpg")
        formated_text, line_num = typo_format_text(key_phrase)
        typo_draw_img(formated_text, line_num, typo_path)

        # 2. 执行拼接
        sd_path = os.path.join(SD_ROOT, f"{img_id}.jpg")
        final_path = os.path.join(RESULT_ROOT, f"{img_id}.jpg")

        if os.path.exists(sd_path):
            vertical_concat(sd_path, typo_path, final_path)
        else:
            print(f"警告: 找不到底图 {sd_path}")

    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f"\n✅ 处理完成！总耗时: {total_time}")
    print(f"最终结果保存在: {RESULT_ROOT}")

if __name__ == "__main__":
    process_all()