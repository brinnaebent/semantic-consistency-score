import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from analysis import get_data

def fig2():
    score_df, lora_df = get_data()

    plt.rc('font', size=13)

    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Plot the boxplot on the first subplot
    axes[0, 0].boxplot([score_df['sdxl_similarity_score'], score_df['pixart_similarity_score']], labels=['SDXL', 'PixArt-α'], patch_artist=True, boxprops=dict(facecolor='lightgray', color='black'), medianprops=dict(color='black'), whiskerprops=dict(color='black'), capprops=dict(color='black'))
    axes[0, 0].set_title('Semantic Consistency Scores: SDXL and PixArt-α')
    axes[0, 0].set_ylabel('Semantic Consistency Score')
    axes[0, 0].set_xlabel('')
    axes[0, 0].grid(True, linestyle='--', alpha=0.7)

    # Plot the KDE plots on the second subplot
    sns.kdeplot(data=score_df['sdxl_similarity_score'], color='gray', linestyle='-', linewidth=2, label='SDXL', ax=axes[0, 1])
    sns.kdeplot(data=score_df['pixart_similarity_score'], color='gray', linestyle='--', linewidth=2, label='PixArt-α', ax=axes[0, 1])
    axes[0, 1].set_title('Score Distributions: SDXL and PixArt-α')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_xlabel('Semantic Consistency Score')
    axes[0, 1].grid(True, linestyle='--', alpha=0.7)
    axes[0, 1].legend()

    axes[1, 0].boxplot([lora_df['sdxl_similarity_score'], lora_df['lora_similarity_score']], labels=['SDXL (base)', 'SDXL (LoRA)'], patch_artist=True, boxprops=dict(facecolor='lightgray', color='black'), medianprops=dict(color='black'), whiskerprops=dict(color='black'), capprops=dict(color='black'))
    axes[1, 0].set_title('Semantic Consistency Scores: Monet Base SDXL and LoRA')
    axes[1, 0].set_ylabel('Semantic Consistency Score')
    axes[1, 0].set_xlabel('')
    axes[1, 0].grid(True, linestyle='--', alpha=0.7)

    sns.kdeplot(data=lora_df['sdxl_similarity_score'], color='gray', linestyle='-', linewidth=2, label='SDXL (base)', ax=axes[1, 1])
    sns.kdeplot(data=lora_df['lora_similarity_score'], color='gray', linestyle='--', linewidth=2, label='SDXL (LoRA)', ax=axes[1, 1])
    axes[1, 1].set_title('Score Distributions: Monet Base SDXL and LoRA')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_xlabel('Semantic Consistency Score')
    axes[1, 1].grid(True, linestyle='--', alpha=0.7)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig('combined_plots.png', dpi=300)  # Save as a high-resolution image
    plt.show()

def main():
    pass

if __name__ == "__main__":
    main()