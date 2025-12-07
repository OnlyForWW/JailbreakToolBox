import pandas as pd
from graphviz import Digraph
import re
import os
import textwrap
import json  # 新增：用于输出 JSON
import yaml


# ========== 辅助函数：智能换行 ==========
def wrap_text(text, width=20):
    if re.match(r'^\d+\.\s*', text):
        match = re.match(r'^(\d+\.\s*)(.+)$', text)
        if match:
            prefix = match.group(1)
            content = match.group(2)
            wrapped = textwrap.fill(content, width=width, break_long_words=False, replace_whitespace=False)
            return prefix + wrapped.replace('\n', '\n' + ' ' * len(prefix))
    return textwrap.fill(text, width=width, break_long_words=False, replace_whitespace=False)


# ========== 配置 ==========
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(BASE_DIR, 'config.yaml'), encoding='utf-8') as f:
    CONFIG = yaml.safe_load(f)

CSV_FILE = CONFIG['path']['output_step_data']
OUTPUT_DIR = CONFIG['path']['output_dir']
FONT_NAME = None  # ✅ 字体控制：None = 默认字体；或设为 'SimHei', 'Microsoft YaHei', 'Arial' 等

# ========== 🎛️ 布局模式选择 ==========
LAYOUT_MODE = 'BOTH'  # 可选: 'LR', 'TB', 'BOTH', 'Z'

# ========== 创建输出目录 ==========
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== 读取数据 ==========
df = pd.read_csv(CSV_FILE)


# ========== 定义生成函数 ==========
def generate_lr_tb_flowchart(question, steps, layout, suffix=''):
    """生成 LR 或 TB 布局的流程图（支持换行 + 高清 + 字体控制）"""
    dot = Digraph(comment=f'Flowchart ({layout})')
    dot.attr(dpi='300')

    # ✅ 设置节点默认样式（仅当 FONT_NAME 非 None 时设置 fontname）
    node_attrs = {
        'shape': 'box',
        'style': 'filled',
        'fillcolor': 'white',
        'fontcolor': 'black',
        'penwidth': '1'
    }
    if FONT_NAME is not None:
        node_attrs['fontname'] = FONT_NAME
    dot.attr('node', **node_attrs)

    if layout == 'LR':
        dot.attr(rankdir='LR')
    else:
        dot.attr(rankdir='TB')

    start_id = 'Start'
    wrapped_question = wrap_text(question, width=25)
    # ✅ 节点字体设置（同上逻辑）
    start_attrs = {'shape': 'oval'}
    if FONT_NAME is not None:
        start_attrs['fontname'] = FONT_NAME
    dot.node(start_id, wrapped_question, **start_attrs)

    prev_id = start_id
    for i, step_text in enumerate(steps):
        node_id = f'Step{i+1}'
        wrapped_step = wrap_text(step_text, width=20)
        step_attrs = {}
        if FONT_NAME is not None:
            step_attrs['fontname'] = FONT_NAME
        dot.node(node_id, wrapped_step, **step_attrs)
        dot.edge(prev_id, node_id)
        prev_id = node_id

    filename = f"{suffix}"
    output_path = os.path.join(OUTPUT_DIR, filename)
    dot.render(output_path, format='png', cleanup=True)
    return f"{filename}.png"


def generate_z_flowchart(question, steps, suffix='', max_per_row=3):
    """生成 Z 字型布局流程图（支持换行 + 高清 + 字体控制）"""
    all_nodes = [question] + steps

    dot = Digraph(comment='Z-Flowchart')
    dot.attr(dpi='300')

    # ✅ 设置节点默认样式
    node_attrs = {
        'shape': 'box',
        'style': 'filled',
        'fillcolor': 'white',
        'fontcolor': 'black',
        'penwidth': '1'
    }
    if FONT_NAME is not None:
        node_attrs['fontname'] = FONT_NAME
    dot.attr('node', **node_attrs)
    dot.attr(rankdir='TB', splines='line')

    # 分组
    grouped_nodes = []
    current_row = []
    for node in all_nodes:
        if len(current_row) < max_per_row:
            current_row.append(node)
        else:
            grouped_nodes.append(current_row)
            current_row = [node]
    if current_row:
        grouped_nodes.append(current_row)

    # 添加节点
    for row_idx, row_nodes in enumerate(grouped_nodes):
        for col_idx, node_text in enumerate(row_nodes):
            node_id = f'n_{row_idx}_{col_idx}'
            wrapped_text = wrap_text(node_text, width=18)
            node_attrs_local = {}
            if FONT_NAME is not None:
                node_attrs_local['fontname'] = FONT_NAME

            if row_idx == 0 and col_idx == 0:
                node_attrs_local['shape'] = 'oval'
                dot.node(node_id, wrapped_text, **node_attrs_local)
            else:
                dot.node(node_id, wrapped_text, **node_attrs_local)

    # Z字连接
    for row_idx, row_nodes in enumerate(grouped_nodes):
        with dot.subgraph() as s:
            s.attr(rank='same')
            if row_idx % 2 == 1:
                for col_idx in range(len(row_nodes) - 1, 0, -1):
                    s.edge(f'n_{row_idx}_{col_idx}',
                           f'n_{row_idx}_{col_idx - 1}',
                           dir='back')
            else:
                for col_idx in range(len(row_nodes) - 1):
                    s.edge(f'n_{row_idx}_{col_idx}',
                           f'n_{row_idx}_{col_idx + 1}')

    # 跨行连接
    for row_idx in range(len(grouped_nodes) - 1):
        last_col_idx = len(grouped_nodes[row_idx]) - 1
        next_first = f'n_{row_idx + 1}_0'
        dot.edge(f'n_{row_idx}_{last_col_idx}', next_first)

    filename = f"{suffix}"
    output_path = os.path.join(OUTPUT_DIR, filename)
    dot.render(output_path, format='png', cleanup=True)
    return f"{filename}.png"


# ========== 主循环 + 收集 JSON 数据 ==========
results_json = []

for idx, row in df.iterrows():
    question = str(row['question']).strip()
    response = str(row['response']).strip()

    response_clean = re.sub(r'</?Steps>', '', response, flags=re.IGNORECASE)
    lines = response_clean.splitlines()

    steps = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if re.match(r'^\d+\.\s*', line):
            steps.append(line)

    if not steps:
        print(f"⚠️  跳过第 {idx+1} 行：无有效步骤")
        continue

    print(f"\n--- 处理第 {idx+1} 行 ---")

    id_num = idx + 1
    img_paths = []

    if LAYOUT_MODE == 'LR':
        png_name = generate_lr_tb_flowchart(question, steps, 'LR', f"{id_num}")
        img_paths.append(os.path.abspath(os.path.join(OUTPUT_DIR, png_name)))

    elif LAYOUT_MODE == 'TB':
        png_name = generate_lr_tb_flowchart(question, steps, 'TB', f"{id_num}")
        img_paths.append(os.path.abspath(os.path.join(OUTPUT_DIR, png_name)))

    elif LAYOUT_MODE == 'BOTH':
        png_lr = generate_lr_tb_flowchart(question, steps, 'LR', f"{id_num}_lr")
        png_tb = generate_lr_tb_flowchart(question, steps, 'TB', f"{id_num}_tb")
        png_z = generate_z_flowchart(question, steps, f"{id_num}_z")
        img_paths.extend([
            os.path.abspath(os.path.join(OUTPUT_DIR, png_lr)),
            os.path.abspath(os.path.join(OUTPUT_DIR, png_tb)),
            os.path.abspath(os.path.join(OUTPUT_DIR, png_z))
        ])

    elif LAYOUT_MODE == 'Z':
        png_z = generate_z_flowchart(question, steps, f"{id_num}_z")
        img_paths.append(os.path.abspath(os.path.join(OUTPUT_DIR, png_z)))

    # 将每张图作为一个条目加入 JSON（同一 question 可能多条）
    for img_path in img_paths:
        results_json.append({
            "id": id_num,
            "question": question,
            "img_path": img_path
        })

    # 打印日志（保持原样）
    if LAYOUT_MODE == 'LR':
        print(f"✅ 已生成（横向）：{png_name}")
    elif LAYOUT_MODE == 'TB':
        print(f"✅ 已生成（纵向）：{png_name}")
    elif LAYOUT_MODE == 'BOTH':
        print(f"✅ 已生成（横向）：{png_lr}")
        print(f"✅ 已生成（纵向）：{png_tb}")
        print(f"✅ 已生成（Z字型）：{png_z}")
    elif LAYOUT_MODE == 'Z':
        print(f"✅ 已生成（Z字型）：{png_z}")

# ========== 保存 JSON 文件 ==========
json_output_path = os.path.join(OUTPUT_DIR, "flowchart_metadata.json")
with open(json_output_path, 'w', encoding='utf-8') as f:
    json.dump(results_json, f, ensure_ascii=False, indent=2)

print(f"\n🎉 所有流程图已生成完毕，保存在：{OUTPUT_DIR}")
print(f"📄 元数据 JSON 已保存至：{os.path.abspath(json_output_path)}")