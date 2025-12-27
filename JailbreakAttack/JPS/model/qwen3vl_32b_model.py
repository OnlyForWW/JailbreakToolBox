import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from .base_model import BaseModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
import torch
from tqdm import tqdm
from PIL import Image
import json
from qwen_vl_utils import process_vision_info

class Qwen3VL32BModel(BaseModel):
    def __init__(self, config):
        self.config = config
        path = config['model']['model_path']
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
        ).eval()

        self.processor = AutoProcessor.from_pretrained(path)
        self.processor.tokenizer.padding_side = "left"
        self.pipeline = {
            'model': self.model,
            'processor': self.processor
        }

        # 固定 image_type 为 'blank'
        if config['data_loader']['image_type'] != 'blank':
            raise ValueError(
                "This model only supports 'blank' image type. Please set 'image_type' to 'blank' in your configuration.")
        self.image_type = 'blank'
        self.inference_save_dir = config['inference_config']['inference_save_dir']

    def test(self, data, adv_img_path, steer_prompt='', save=True):
        attack_type = self.config['attack_config']['attack_type']
        if attack_type != 'full':
            raise ValueError(f"This model only supports 'full' attack type. Found '{attack_type}' instead.")

        image_height = self.config['data_loader']['image_height']
        image_width = self.config['data_loader']['image_width']

        # 仅支持 attack_type == 'full'
        adv_image = Image.open(adv_img_path).convert('RGB').resize((image_width, image_height))

        # 仅支持 blank 图像：问题 = steer_prompt + origin_question
        question_list = [steer_prompt + data_dict['origin_question'] for data_dict in data]
        type_list = [data_dict['type'] for data_dict in data]

        # 所有样本使用同一张对抗图像（full attack）
        image_list = [adv_image] * len(question_list)

        outputs = self.inference(
            question_list,
            image_list,
            max_new_tokens=self.config['inference_config']['max_new_tokens'],
            batch_size=self.config['inference_config']['batch_size']
        )

        assert len(question_list) == len(outputs) == len(type_list)

        if not save:
            return outputs, type_list

        # 构建保存路径
        relative_path = os.path.relpath(adv_img_path, self.config['attack_config']['adv_save_dir'])
        filename = os.path.splitext(relative_path)[0]
        result_file = os.path.join(self.inference_save_dir, filename + '.json')
        parts = result_file.split('/')
        parts[-3] = self.config['data_loader']['dataset_name']
        result_file = '/'.join(parts)
        os.makedirs(os.path.dirname(result_file), exist_ok=True)

        results = [
            {"question": q, "answer": a, "type": t}
            for q, a, t in zip(question_list, outputs, type_list)
        ]
        with open(result_file, 'w', encoding='utf-8') as file:
            json.dump(results, file, ensure_ascii=False, indent=4)
        return result_file

    def inference(self, texts, images, max_new_tokens=1024, batch_size=5):
        assert isinstance(texts, list)
        assert isinstance(images, list)
        assert len(texts) == len(images)

        generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False)
        messages = [self.get_prompt(text, image) for text, image in zip(texts, images)]

        responses = []
        for i in tqdm(range(0, len(messages), batch_size), file=self.config['tqdm_out']):
            batch_messages = messages[i:i + batch_size]
            batch_texts = [
                self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in batch_messages
            ]
            image_inputs, video_inputs = process_vision_info(batch_messages, image_patch_size=16)
            inputs = self.processor(
                text=batch_texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")
            generated_ids = self.model.generate(**inputs, **generation_config)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            batch_output_texts = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            responses.extend(batch_output_texts)
        return responses

    def get_prompt(self, input_text, image):
        # 仅支持带图像的输入（full attack always has image）
        return [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": input_text},
            ],
        }]