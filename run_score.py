import os
import pandas as pd
from semantic_score import embed_images, calculate_similarity

def pixart_sdxl_eval(save_path='results/pixart_sdxl_eval_df'):
    PROMPTS = 'resources/prompts.txt'
    SEEDS = 'resources/random_seeds.txt'
    SDXL_RESULTS_PATH = 'results/sdxl'
    PIXART_RESULTS_PATH = 'results/pixart'

    with open(PROMPTS, 'r') as file:
        prompts = [line.rstrip('\n') for line in file][0:100]

    seeds = []
    with open(SEEDS, 'r') as file:
        for line in file:
            seeds.append(int(line.strip()))

    score_df = pd.DataFrame()
    for prompt in prompts:
        sdxl_files = os.listdir(f'{SDXL_RESULTS_PATH }/{prompt}')
        pixart_files = os.listdir(f'{PIXART_RESULTS_PATH}/{prompt}')
        sdxl_image_paths, pixart_image_paths = [],[]
        for seed in seeds[0:20]:
            sdxl_image_paths.append(f'{SDXL_RESULTS_PATH }/{prompt}/{seed}.jpg')
            pixart_image_paths.append(f'{PIXART_RESULTS_PATH}/{prompt}/{seed}.jpg')

        sdxl_embeddings = embed_images(sdxl_image_paths)
        pixart_embeddings = embed_images(pixart_image_paths)
        sdxl_similarity_score = calculate_similarity(sdxl_embeddings)
        pixart_similarity_score = calculate_similarity(pixart_embeddings)

        score_df = score_df.append({
            'prompt': prompt,
            'sdxl_similarity_score': sdxl_similarity_score,
            'pixart_similarity_score': pixart_similarity_score
        }, ignore_index=True)

    score_df.to_csv(save_path, mode='a', header=False, index=False)


def lora_eval(save_path='results/lora_sdxl_eval_df'):
    SDXL_PROMPTS = 'resources/prompts_monet_sdxl.txt'
    LORA_PROMPTS = 'resources/prompts_monet_lora.txt'
    SEEDS = 'resources/random_seeds.txt'
    SDXL_RESULTS_PATH = 'results/sdxl'
    
    with open(SDXL_PROMPTS, 'r') as file:
        sdxl_prompts = [line.rstrip('\n') for line in file]

    with open(LORA_PROMPTS, 'r') as file:
        lora_prompts = [line.rstrip('\n') for line in file]

    seeds = []
    with open(SEEDS, 'r') as file:
        for line in file:
            seeds.append(int(line.strip()))

    lora_df = pd.DataFrame()
    for idx, prompt in enumerate(sdxl_prompts):
        sdxl_files = os.listdir(f'{SDXL_RESULTS_PATH}/{prompt}')
        lora_files = os.listdir(f'{SDXL_RESULTS_PATH}/{lora_prompts[idx]}')
        sdxl_image_paths, lora_image_paths, base_image_paths = [],[],[]
        for seed in seeds[0:20]:
            sdxl_image_paths.append(f'{SDXL_RESULTS_PATH}/{prompt}/{seed}.jpg')
            lora_image_paths.append(f'{SDXL_RESULTS_PATH}/{lora_prompts[idx]}/{seed}.jpg')

        sdxl_embeddings = embed_images(sdxl_image_paths)
        lora_embeddings = embed_images(lora_image_paths)
        sdxl_similarity_score = calculate_similarity(sdxl_embeddings)
        lora_similarity_score = calculate_similarity(lora_embeddings)

        lora_df = lora_df.append({
            'prompt': prompt,
            'sdxl_similarity_score': sdxl_similarity_score,
            'lora_similarity_score': lora_similarity_score,
        }, ignore_index=True)

def main():
    pass

if __name__ == "__main__":
    main()
