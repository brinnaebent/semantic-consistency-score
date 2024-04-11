import os
from tqdm import tqdm
from generate_images import run_pixart, run_sdxl, run_sdxl_lora

def create_directory(directory_name, path):
    """
    Creates a new directory if it doesn't exist.
    
    Parameters:
    - directory_name: Name of the directory to create.
    - path: Parent path where the directory will be created.
    
    Returns:
    - The full path to the created directory.
    """
    directory_path = os.path.join(path, directory_name)

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    return directory_path

def run_generation_loop(prompts, seeds, model, save_path):
    """
    Generates images for each prompt and seed, saving them to a specified path.
    
    Parameters:
    - prompts: A list of prompts for image generation.
    - seeds: A list of seeds for image generation.
    - model: The model to use for image generation ('pixart', 'sdxl', or 'lora').
    - save_path: The path where generated images will be saved.
    """
    for prompt in tqdm(prompts, desc="Generating images."):
        path_to_save = create_directory(prompt, save_path)
        for seed in tqdm(seeds[:20], desc=f"Generating images for '{prompt}'"):
            if model == 'pixart':
                run_pixart(prompt, seed, path_to_save)
            elif model == 'sdxl':
                run_sdxl(prompt, seed, path_to_save)
            elif model == 'lora':
                run_sdxl_lora(prompt, seed, path_to_save)

def main(model, save_path, prompts='../resources/prompts.txt'):
    """
    Main function to execute the script.
    Reads prompts and seeds from files, then starts the image generation loop.
    """
    PROMPTS = prompts
    SEEDS = '../resources/random_seeds.txt'

    # Read prompts from a file, limited to the first 100.
    with open(PROMPTS, 'r') as file:
        prompts = [line.rstrip('\n') for line in file][:100]

    # Read seeds from a file, converting each to an integer.
    seeds = []
    with open(SEEDS, 'r') as file:
        seeds = [int(line.strip()) for line in file]

    # Start the image generation loop.
    run_generation_loop(prompts, seeds, model, save_path)

if __name__ == "__main__":
    main()
