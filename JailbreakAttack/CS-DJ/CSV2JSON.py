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

df.insert(0, 'id', range(1, 1 + len(df)))

os.makedirs(CONFIG['path']['jailbreak_folder_path'], exist_ok=True)

josn_save_path = os.path.join(CONFIG['path']['jailbreak_folder_path'], f'{json_name}.json')

df.to_json(josn_save_path, orient='records', force_ascii=False, indent=4)

print("Data process deal!")