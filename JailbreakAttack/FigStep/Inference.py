import json
import os
import yaml
import base64
import asyncio
from openai import AsyncOpenAI  # 注意引入的是 AsyncOpenAI
from tqdm.asyncio import tqdm  # 使用异步兼容的 tqdm

# ================= 路径与配置加载 =================
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_config():
    config_path = os.path.join(CURRENT_SCRIPT_DIR, 'config.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_abs_path(rel_path):
    if os.path.isabs(rel_path):
        return rel_path
    return os.path.abspath(os.path.join(CURRENT_SCRIPT_DIR, rel_path))


CONFIG = load_config()

# 路径变量
INPUT_JSON = get_abs_path(CONFIG['paths']['output_json'])
IMAGE_FOLDER = get_abs_path(CONFIG['paths']['output_image_folder'])
OUTPUT_RESULT = get_abs_path(CONFIG['inference']['output_result_file'])

# 模型参数
VLLM_BASE_URL = CONFIG['inference']['base_url']
VLLM_API_KEY = CONFIG['inference']['api_key']
MODEL_NAME = CONFIG['inference']['model_name']
INCITEMENT_PROMPT = CONFIG['inference']['incitement_prompt']
CONCURRENCY_LIMIT = CONFIG['inference'].get('concurrency_limit', 10)
MAX_RETRIES = CONFIG['inference'].get('max_retries', 3)  # 从配置读取重试次数，默认3次
MAX_TOKENS = CONFIG['inference']['max_new_tokens']
# temperature
TEMPERATURE = CONFIG['inference']['temperature']


# ================= 辅助函数 =================

def encode_image(image_path):
    """将图片转换为 Base64 字符串 (同步IO操作)"""
    if not os.path.exists(image_path):
        return None
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def save_results(results):
    """保存结果到 JSON"""
    # 按 ID 排序
    sorted_results = sorted(results, key=lambda x: x['id'])

    os.makedirs(os.path.dirname(OUTPUT_RESULT), exist_ok=True)
    with open(OUTPUT_RESULT, 'w', encoding='utf-8') as f:
        json.dump(sorted_results, f, ensure_ascii=False, indent=4)


# ================= 核心异步逻辑 =================

async def process_single_item(client, item, semaphore):
    """处理单条数据的异步函数（包含重试机制）"""
    async with semaphore:  # 获取信号量，控制并发
        item_id = item.get('id')
        original_q = item.get('original_question')

        # 准备返回结构（只保留您要求的字段）
        final_result = {
            "id": item_id,
            "original_question": original_q,
            "response": None
        }

        image_filename = f"{item_id}.png"
        image_path = os.path.join(IMAGE_FOLDER, image_filename)

        # 编码图片
        base64_image = encode_image(image_path)

        if not base64_image:
            final_result["response"] = "Error: Image file missing"
            return final_result

        # === 重试循环 ===
        for attempt in range(MAX_RETRIES):
            try:
                # 异步调用 API
                response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": INCITEMENT_PROMPT},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image}"
                                    },
                                },
                            ],
                        }
                    ],
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE
                )

                # 成功获取结果
                model_output = response.choices[0].message.content
                final_result["response"] = model_output
                return final_result  # 成功则直接返回

            except Exception as e:
                # 如果是最后一次尝试，则记录错误并退出
                if attempt == MAX_RETRIES - 1:
                    error_msg = f"Failed after {MAX_RETRIES} attempts. Last error: {str(e)}"
                    print(f"\n[ID {item_id}] 最终失败: {error_msg}")
                    final_result["response"] = error_msg
                    return final_result

                # 如果不是最后一次，打印警告并等待后重试
                wait_time = 2 * (attempt + 1)  # 简单的退避策略：2s, 4s, 6s...
                # print(f"\n[ID {item_id}] 第 {attempt+1} 次尝试失败 ({str(e)})，{wait_time}秒后重试...")
                await asyncio.sleep(wait_time)

        return final_result


async def main():
    # 1. 初始化异步客户端
    client = AsyncOpenAI(
        base_url=VLLM_BASE_URL,
        api_key=VLLM_API_KEY,
    )
    print(f"已连接 vLLM (Async): {VLLM_BASE_URL}")
    print(f"并发限制: {CONCURRENCY_LIMIT} | 最大重试次数: {MAX_RETRIES}")

    # 2. 读取数据
    if not os.path.exists(INPUT_JSON):
        print("未找到输入文件。")
        return

    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    # 3. 创建信号量和任务列表
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    tasks = []

    print(f"开始构造 {len(data_list)} 个任务...")
    for item in data_list:
        task = process_single_item(client, item, semaphore)
        tasks.append(task)

    # 4. 执行并收集结果
    results = []

    # 使用 asyncio.as_completed 配合 tqdm 显示实时进度
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Inferencing"):
        result = await f
        results.append(result)

        # 每完成 10 条保存一次，防止意外中断
        if len(results) % 10 == 0:
            save_results(results)

    # 5. 最终保存
    save_results(results)
    print(f"\n全部完成！结果已保存至: {OUTPUT_RESULT}")

    # 关闭客户端
    await client.close()

if __name__ == "__main__":
    asyncio.run(main())