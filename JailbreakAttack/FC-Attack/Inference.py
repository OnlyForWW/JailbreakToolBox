import json
import os
import base64
import asyncio
from openai import AsyncOpenAI
from PIL import Image
from io import BytesIO
from tqdm.asyncio import tqdm_asyncio
import yaml

# ==================== 全局配置 ====================

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(BASE_DIR, 'config.yaml'), encoding='utf-8') as f:
    CONFIG = yaml.safe_load(f)

INPUT_JSON_PATH = CONFIG['path']['metadata']
OUTPUT_JSON_PATH = CONFIG['inference']['output_result_file']
os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)

VLLM_API_BASE = CONFIG['inference']['base_url']
MODEL_NAME = CONFIG['inference']['model_name']
MAX_CONCURRENT = CONFIG['inference']['concurrency_limit']
MAX_RETRIES = CONFIG['inference']['max_retries']
PROMPT_TEMPLATE = CONFIG['inference']['incitement_prompt']
SYSTEM_PROMPT = CONFIG['inference']['system_prompt']
MAX_TOKENS = CONFIG['inference']['max_new_tokens']
TEMPERATURE = CONFIG['inference']['temperature']

# ==================== 工具函数 ====================

def encode_image_to_base64(image_path: str) -> str:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像不存在: {image_path}")
    with Image.open(image_path) as img:
        # 转为 RGB（防止 RGBA 问题）
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")

# ==================== 推理函数 ====================

async def run_single_inference(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    item: dict
) -> dict:
    idx = item["id"]
    question = item["question"]
    img_path = item["img_path"]

    async with semaphore:
        try:
            base64_image = encode_image_to_base64(img_path)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",
                "content": [
                    {"type": "text", "text": PROMPT_TEMPLATE},
                    {"type": "image_url", "image_url": {"url": base64_image}}
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
                        "question": question,
                        "img_path": img_path,
                        "response": response.choices[0].message.content.strip()
                    }
                except Exception as e:
                    if attempt == MAX_RETRIES - 1:
                        return {
                            "id": idx,
                            "question": question,
                            "img_path": img_path,
                            "response": f"[ERROR] Retry {MAX_RETRIES} failed: {str(e)}"
                        }
                    await asyncio.sleep(1 * (attempt + 1))
        except Exception as e:
            return {
                "id": idx,
                "question": question,
                "img_path": img_path,
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
    for item in data:
        # 确保必要字段存在
        if not all(k in item for k in ["id", "question", "img_path"]):
            # 构造缺失项（尽量保留 id）
            idx = item.get("id", 0)
            tasks.append(asyncio.create_task(asyncio.sleep(0, {
                "id": idx,
                "question": item.get("question", ""),
                "img_path": item.get("img_path", ""),
                "response": "[SKIPPED] Missing required fields"
            })))
        else:
            tasks.append(run_single_inference(client, semaphore, item))

    print(f"📊 共 {len(data)} 条数据，最大并发: {MAX_CONCURRENT}")
    results = await tqdm_asyncio.gather(*tasks, desc="🧠 推理中")

    # 按 id 排序（可选，因输入已有序）
    results.sort(key=lambda x: x["id"])

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 完成！结果已保存至 {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    asyncio.run(main())