import pandas as pd
import yaml
import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(BASE_DIR, 'config.yaml'), encoding='utf-8') as f:
    CONFIG = yaml.safe_load(f)

CONFIG['path'] = {k: os.path.join(BASE_DIR, v) for k, v in CONFIG['path'].items()}

json_name = CONFIG['origin_data'].split('/')[-1].split('.')[0]

df = pd.read_csv(CONFIG['origin_data'])

df.rename(columns={'question': 'instruction'}, inplace=True)

# 期望的 ID 序列：从 1 到 len(df)
expected_id = list(range(1, len(df) + 1))

# 判断是否需要重置 id 列
should_reset_id = True
if 'id' in df.columns:
    try:
        # 尝试将 id 列转换为整数列表（处理可能的 float 或字符串）
        current_id = pd.to_numeric(df['id'], errors='coerce').astype('Int64').tolist()
        # 检查是否完全匹配 expected_id（包括顺序和值）
        if current_id == expected_id:
            should_reset_id = False
    except Exception:
        # 如果转换失败，说明 id 列格式异常，需要重置
        pass

# 如果不符合要求，则删除旧 id（如果存在）并插入新 id
if should_reset_id:
    if 'id' in df.columns:
        df.drop(columns=['id'], inplace=True)
    df.insert(0, 'id', expected_id)

os.makedirs(CONFIG['path']['jailbreak_folder_path'], exist_ok=True)

json_save_path = os.path.join(CONFIG['path']['jailbreak_folder_path'], f'{json_name}.json')

df.to_json(json_save_path, orient='records', force_ascii=False, indent=4)

print("Data process deal!")