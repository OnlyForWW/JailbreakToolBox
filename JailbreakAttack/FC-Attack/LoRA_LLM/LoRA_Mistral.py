# 加载库
from datasets import Dataset
import pandas as pd
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          DataCollatorForSeq2Seq,
                          TrainingArguments,
                          Trainer,
                          GenerationConfig)
from peft import LoraConfig, TaskType, get_peft_model

DATA_PATH = 'step_data.json'
MODEL_PATH = '/home/wangjy/data/LLM/AI-ModelScope/Mistral-7B-Instruct-v0.1'

LoRA_R = 16
LoRA_ALPHA = 64
LoRA_DROPOUT = 0.05
BIAS = 'none'

# 数据集加载
df = pd.read_json(DATA_PATH)
ds = Dataset.from_pandas(df)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
# <s>[INST] 你好呀 [/INST]你好，你有什么事情要问我吗？</s>
def process_func(example):
    MAX_LENGTH = 512
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(f"<s>[INST]{example['query']}[/INST]", add_special_tokens=False)
    response = tokenizer(f"{example['steps']}</s>", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [-100]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

tokenized_id = ds.map(process_func, remove_columns=ds.column_names)

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="cuda", torch_dtype='auto')
model.enable_input_require_grads()
print(f">>>>>>>>>> Model Dtype{model.dtype}. >>>>>>>>>>")

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=LoRA_R,
    lora_alpha=LoRA_ALPHA,
    lora_dropout=LoRA_DROPOUT,
    bias=BIAS
)

model = get_peft_model(model, config)

print(">>>>>>>>>> Training Parameters: ")
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir="./Mistral-7B-Instruct-v0.1_lora_finetuned",           # 模型保存路径
    num_train_epochs=2,                      # ← 论文：2 epochs
    per_device_train_batch_size=8,           # ← 单卡直接=8（global batch=8）
    gradient_accumulation_steps=1,           # 单卡显存够就不累积
    learning_rate=1e-5,                      # ← 论文：1e-5
    weight_decay=1e-5,                       # ← 论文：1e-5
    lr_scheduler_type="cosine",              # 推荐 cosine 衰减
    warmup_ratio=0.03,                       # 前 3% steps 热身
    logging_steps=10,                        # 每10步打印一次loss
    save_strategy="epoch",                   # 每个epoch保存一次
    save_total_limit=2,                      # 最多保留2个checkpoint
    bf16=True,                               # 若GPU支持（A100/3090/4090等）
    optim="adamw_torch",                     # 标准优化器
    gradient_checkpointing=True,             # ✅ 节省显存，强烈建议开启
    remove_unused_columns=False,             # ✅ 必须！否则 labels 会被删
    report_to="none",                        # 不用 wandb/tensorboard
    seed=42,                                 # 保证复现性
    dataloader_num_workers=4,                # 加速数据加载
    dataloader_pin_memory=True,              # 加速数据传输到GPU
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,              # 动态 padding
    return_tensors="pt",       # 返回 PyTorch tensor（默认就是）
    pad_to_multiple_of=8,      # 👈 对齐到 8 的倍数，提升 bf16/amp 效率（L40S 友好！）
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_id,
    data_collator=data_collator
)

trainer.train()

final_lora_path = "./Mistral-7B-Instruct-v0.1_lora_final"
trainer.model.save_pretrained(final_lora_path)
tokenizer.save_pretrained(final_lora_path)