import json
import os
import yaml
import textwrap
from PIL import Image, ImageFont, ImageDraw

# ================= 路径与配置加载 =================
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_SCRIPT_DIR))
LOCAL_CONFIG_PATH = os.path.join(CURRENT_SCRIPT_DIR, 'config.yaml')


def load_config():
    if not os.path.exists(LOCAL_CONFIG_PATH):
        raise FileNotFoundError(f"配置文件未找到: {LOCAL_CONFIG_PATH}")
    with open(LOCAL_CONFIG_PATH, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


CONFIG = load_config()


# 解析绝对路径
def get_abs_path(rel_path):
    if os.path.isabs(rel_path):
        return rel_path
    return os.path.abspath(os.path.join(CURRENT_SCRIPT_DIR, rel_path))


FONT_PATH = get_abs_path(CONFIG['paths']['font_path'])
INPUT_JSON = get_abs_path(CONFIG['paths']['output_json'])  # 这里读取上一步生成的JSON
OUTPUT_FOLDER = get_abs_path(CONFIG['paths']['output_image_folder'])

IMG_PARAMS = CONFIG['image_generation']


# ================= 核心绘图逻辑 =================

def get_font():
    """加载字体，如果指定字体不存在则尝试加载默认字体"""
    try:
        return ImageFont.truetype(FONT_PATH, IMG_PARAMS['font_size'])
    except OSError:
        print(f"警告: 未找到字体文件 {FONT_PATH}，将使用系统默认字体。排版效果可能与论文不一致。")
        return ImageFont.load_default()


def wrap_text(text):
    """文本自动换行"""
    return textwrap.fill(text, width=IMG_PARAMS['wrap_width'])


def text_step_by_step(text: str, steps=3):
    """
    构造 FigStep 所需的文本格式：
    标题 + 编号列表 (1. \n 2. \n 3.)

    """
    text = text.removesuffix("\n")
    # 先进行换行处理
    text = wrap_text(text)
    # 添加 1, 2, 3 序号
    for idx in range(1, steps + 1):
        text += f"\n{idx}. "
    return text


def text_to_image(text: str):
    """绘制图像"""
    font = get_font()

    # 创建白色背景图片
    # 论文指定背景为白色 (#FFFFFF)
    w, h = IMG_PARAMS['image_size']
    im = Image.new("RGB", (w, h), "#FFFFFF")
    dr = ImageDraw.Draw(im)

    # 绘制黑色文字
    x, y = IMG_PARAMS['xy_offset']
    draw_kwargs = {
        "xy": (x, y),
        "text": text,
        "spacing": IMG_PARAMS['line_spacing'],
        "font": font,
        "fill": "#000000"
    }

    dr.text(**draw_kwargs)
    return im


def main():
    # 1. 确保输出目录存在
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"创建输出目录: {OUTPUT_FOLDER}")

    # 2. 读取 JSON 数据
    if not os.path.exists(INPUT_JSON):
        print(f"错误: 未找到输入文件 {INPUT_JSON}，请先运行改写脚本。")
        return

    try:
        with open(INPUT_JSON, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print("错误: JSON 文件格式不正确。")
        return

    print(f"开始生成图像，共 {len(data)} 条数据...")

    success_count = 0

    for item in data:
        # 获取 ID 和 改写后的文本
        item_id = item.get('id')
        paraphrased_text = item.get('paraphrased_question')

        # 如果改写失败或为空，则跳过
        if not paraphrased_text or paraphrased_text == "FAILED_TO_GENERATE":
            print(f"跳过 ID {item_id}: 无有效改写文本")
            continue

        try:
            # 1. 格式化文本 (添加换行和序号)
            processed_text = text_step_by_step(paraphrased_text, steps=3)

            # 2. 绘制图像
            image = text_to_image(processed_text)

            # 3. 保存图像
            # 使用 ID 作为文件名，方便后续索引
            filename = f"{item_id}.png"
            output_path = os.path.join(OUTPUT_FOLDER, filename)
            image.save(output_path)

            success_count += 1
            if success_count % 10 == 0:
                print(f"已生成 {success_count} 张图片...")

        except Exception as e:
            print(f"处理 ID {item_id} 时发生错误: {e}")

    print(f"处理完成！成功生成 {success_count} 张图片，保存至 {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()