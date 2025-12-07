import json
import os
import base64
import asyncio
import yaml
from openai import AsyncOpenAI
from PIL import Image
from io import BytesIO
from tqdm.asyncio import tqdm_asyncio

# ==================== 全局配置 ====================

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(BASE_DIR, 'config.yaml'), encoding='utf-8') as f:
    CONFIG = yaml.safe_load(f)

CONFIG['path'] = {k: os.path.join(BASE_DIR, v) for k, v in CONFIG['path'].items()}

INPUT_JSON_PATH = next(os.path.join(CONFIG['path']['jailbreak_images_question_map_save_path'], f) 
                       for f in os.listdir(CONFIG['path']['jailbreak_images_question_map_save_path']) 
                       if f.lower().endswith('.json'))
OUTPUT_JSON_PATH = CONFIG['inference']['output_result_file']
os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)

VLLM_API_BASE = CONFIG['inference']['base_url']
MODEL_NAME = CONFIG['inference']['model_name']
MAX_CONCURRENT = CONFIG['inference']['concurrency_limit']
MAX_RETRIES = CONFIG['inference']['max_retries']
PROMPT_TEMPLATE = CONFIG['inference']['incitement_prompt']
MAX_TOKENS = CONFIG['inference']['max_new_tokens']
TEMPERATURE = CONFIG['inference']['temperature']

# ==================== 工具函数 ====================

def encode_image_to_base64(image_path: str) -> str:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像不存在: {image_path}")
    with Image.open(image_path) as img:
        buffer = BytesIO()
        # 强制转为 PNG 以确保兼容性
        img.save(buffer, format="PNG")
        # --- 修改点在这里 ---
        # 原代码: return "image/png;base64," + ...
        # 修改后: 必须包含 "data:" 前缀
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")

# ==================== 推理函数 ====================

async def run_single_inference(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    image_path: str,
    instruction: str,
    idx: int
) -> dict:
    async with semaphore:
        try:
            base64_image = encode_image_to_base64(image_path)
            # 使用全局模板生成最终 prompt
            user_prompt = PROMPT_TEMPLATE.format(instruction=instruction)

            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"{base64_image}"}}
                ]
            }]

            for attempt in range(MAX_RETRIES):
                try:
                    response = await client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        max_tokens=MAX_TOKENS,
                        temperature=TEMPERATURE
                    )
                    return {
                        "id": idx,
                        "original_instruction": instruction,
                        "response": response.choices[0].message.content.strip()
                    }
                except Exception as e:
                    if attempt == MAX_RETRIES - 1:
                        return {
                            "id": idx,
                            "original_instruction": instruction,
                            "response": f"[ERROR] Retry {MAX_RETRIES} failed: {str(e)}"
                        }
                    await asyncio.sleep(1 * (attempt + 1))
        except Exception as e:
            return {
                "id": idx,
                "original_instruction": instruction,
                "response": f"[CRITICAL ERROR] {str(e)}"
            }

# ==================== 主函数 ====================

async def main():
    client = AsyncOpenAI(base_url=VLLM_API_BASE, api_key="token-abc123")
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    with open(INPUT_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("输入 JSON 必须是列表")

    tasks = []
    for idx, item in enumerate(data, start=1):
        image_path = item.get("image_saved_path")
        instruction = item.get("original_instruction")
        if not image_path or not instruction:
            result = {
                "id": idx,
                "original_instruction": instruction or "",
                "response": "[SKIPPED] Missing image_path or original_instruction"
            }
            tasks.append(asyncio.create_task(asyncio.sleep(0, result)))
        else:
            tasks.append(
                run_single_inference(client, semaphore, image_path, instruction, idx)
            )

    print(f"📊 共 {len(data)} 条数据，最大并发: {MAX_CONCURRENT}")
    results = await tqdm_asyncio.gather(*tasks, desc="🧠 推理中")

    results.sort(key=lambda x: x["id"])  # 保险起见

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 完成！结果已保存至 {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    asyncio.run(main())