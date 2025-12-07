import os
import json
import torch
import yaml
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

def main(jailbreak_folder_path, save_embedding_path, seed, num_images, save_map_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    file_list = os.listdir(jailbreak_folder_path)
    jailbreak_question_list = []

    for file in file_list:
        with open(os.path.join(jailbreak_folder_path, file), 'r', encoding='utf-8') as f:
            data = json.load(f)
        jailbreak_question_list.extend([item['instruction'] for item in data])

    with open(os.path.join(save_embedding_path, f'map_seed_{seed}_num_{num_images}.json'), 'r', encoding='utf-8') as f:
        embeding_data = json.load(f)

    image_embeddings = [item['img_emb'] for item in embeding_data]
    image_paths = [item['img_path'] for item in embeding_data]

    image_embeddings = torch.tensor(image_embeddings).to(device)
    model = SentenceTransformer('clip-ViT-L-14').to(device)

    results = {}
    for jailbreak_question in tqdm(jailbreak_question_list, desc="Processing questions"):
        max_distance_embedding_list = []
        selected_image_list = []

        text_emb = model.encode(jailbreak_question, convert_to_tensor=True).to(device)
        max_distance_embedding_list.append(text_emb)

        for _ in range(15):
            combined_emb = torch.vstack(max_distance_embedding_list)
            cos_scores = util.cos_sim(combined_emb, image_embeddings).mean(dim=0) 

            min_score, min_index = torch.min(cos_scores, dim=0)
            selected_image_list.append(image_paths[int(min_index)])
            max_distance_embedding_list.append(image_embeddings[min_index])

        results[jailbreak_question] = selected_image_list

    if not os.path.exists(save_map_path):
        os.makedirs(save_map_path)
    with open(os.path.join(save_map_path, f'distraction_image_map_seed_{seed}_num_{num_images}.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Results saved to {os.path.join(save_map_path, f'distraction_image_map_seed_{seed}_num_{num_images}.json')}")

if __name__ == "__main__":
    
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))

    with open(os.path.join(BASE_DIR, 'config.yaml'), encoding='utf-8') as f:
        CONFIG = yaml.safe_load(f)

    CONFIG['path'] = {k: os.path.join(BASE_DIR, v) for k, v in CONFIG['path'].items()}

    main(
        jailbreak_folder_path=CONFIG['path']['jailbreak_folder_path'],
        save_embedding_path=CONFIG['path']['save_embedding_path'],
        seed=CONFIG['seed'],
        num_images=CONFIG['num_images'],
        save_map_path=CONFIG['path']['save_map_path']
    )
