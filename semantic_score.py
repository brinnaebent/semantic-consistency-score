import os
import torch
import clip
from tqdm import tqdm
from PIL import Image
from typing import List
import pandas as pd

def embed_images(image_paths: List[str]) -> torch.Tensor:
    """
    Embeds a list of image paths using CLIP.

    Args:
        image_paths (List[str]): A list of image paths.

    Returns:
        torch.Tensor: A tensor containing the image embeddings.
    """
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    images = [preprocess(Image.open(path)).unsqueeze(0) for path in image_paths]
    images = torch.cat(images, dim=0).to(device)
    with torch.no_grad():
        embeddings = model.encode_image(images)
    return embeddings

def calculate_similarity(embeddings: torch.Tensor) -> float:
    """
    Calculates the mean pairwise cosine similarity between image embeddings.

    Args:
        embeddings (torch.Tensor): A tensor containing the image embeddings.

    Returns:
        float: The mean pairwise cosine similarity scaled between 0 and 100.
    """
    cosine_similarities = []
    for i in tqdm(range(len(embeddings)), desc="Calculating pairwise similarities"):
        for j in range(i + 1, len(embeddings)):
            cos_sim = torch.cosine_similarity(embeddings[i], embeddings[j], dim=0)
            scaled_similarity = max(100 * cos_sim.item(), 0)
            cosine_similarities.append(scaled_similarity)
    return torch.tensor(cosine_similarities).mean().item()


def main():
    pass

if __name__ == "__main__":
    main()
