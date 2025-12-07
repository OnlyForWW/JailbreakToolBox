from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel
import pandas as pd
import os
import yaml

# ==============================
# 参数配置区（请在此处统一修改）
# ==============================

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(BASE_DIR, 'config.yaml'), encoding='utf-8') as f:
    CONFIG = yaml.safe_load(f)


# 模型与 LoRA 路径
MODEL_PATH = CONFIG['path']['model']
LORA_PATH = CONFIG['path']['LoRA']

# 文件路径
PROMPT_TEMPLATE = CONFIG['Gen_Step_Prompt']
INPUT_CSV_FILE = CONFIG['path']['input_data']
OUTPUT_CSV_FILE = CONFIG['path']['output_step_data']

os.makedirs(os.path.dirname(OUTPUT_CSV_FILE), exist_ok=True)

# 生成参数
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.0

# 其他配置
QUESTION_COLUMN = 'question'  # CSV 中包含问题的列名
DEVICE = 'cuda'

# ==============================
# 代码主体（一般无需修改）
# ==============================



# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, 
                                          trust_remote_code=True)

# 加载基础模型
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map=DEVICE,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).eval()

# 加载 LoRA 权重
model = PeftModel.from_pretrained(model, model_id=LORA_PATH)

# # 从文件加载 prompt 模板
# with open(PROMPT_TEMPLATE_FILE, "r", encoding="utf-8") as f:
#     prompt_template = f.read()

# 读取问题 CSV 文件
df = pd.read_csv(INPUT_CSV_FILE)

# 创建一个列表用于保存结果
results = []

# 遍历每个问题
for idx, row in df.iterrows():
    actual_question = row[QUESTION_COLUMN]

    # 替换模板中的占位符
    prompt = PROMPT_TEMPLATE.replace("{{QUESTION}}", actual_question)

    # 构造消息
    messages = [{"role": "user", "content": prompt}]

    # 使用 tokenizer 处理输入
    input_ids_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([input_ids_str], return_tensors="pt").to(DEVICE)

    # 生成回答
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        pad_token_id=tokenizer.eos_token_id
    )

    # 移除输入部分，只保留生成的回答
    generated_ids = [
        output_ids[len(input_ids):] 
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # 解码生成的回答
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # 保存结果
    results.append({
        'question': actual_question,
        'response': response
    })

    # 打印进度
    print(f"Processed {idx + 1}/{len(df)}: {actual_question[:50]}...")

# 转换为 DataFrame 并保存到 CSV
output_df = pd.DataFrame(results)
output_df.to_csv(OUTPUT_CSV_FILE, index=False, encoding="utf-8")

print(f"✅ 所有问题已处理完毕，结果已保存到 {OUTPUT_CSV_FILE}")