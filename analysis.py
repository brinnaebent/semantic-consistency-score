import pandas as pd
import numpy as np
from scipy.stats import kstest
from scipy.stats import ks_2samp, wilcoxon

def get_data():
    SCORE_DF_PATH = 'results/pixart_sdxl_eval_df'
    LORA_DF_PATH = 'results/lora_sdxl_eval_df'

    score_df = pd.read_csv(SCORE_DF_PATH)
    lora_df = pd.read_csv(LORA_DF_PATH)

    return score_df, lora_df

def print_basic_stats(score_df, lora_df):
    print(f'Mean SDXL: {np.mean(score_df["sdxl_similarity_score"])}')
    print(f'Median SDXL: {np.median(score_df["sdxl_similarity_score"])}')
    print(f'Stdev SDXL: {np.std(score_df["sdxl_similarity_score"])}')
    print('######')
    print(f'Mean pixart: {np.mean(score_df["pixart_similarity_score"])}')
    print(f'Median pixart: {np.median(score_df["pixart_similarity_score"])}')
    print(f'Stdev pixart: {np.std(score_df["pixart_similarity_score"])}')
    print('######')
    print(f'Mean SDXL: {np.mean(lora_df["sdxl_similarity_score"])}')
    print(f'Median SDXL: {np.median(lora_df["sdxl_similarity_score"])}')
    print(f'Stdev SDXL: {np.std(lora_df["sdxl_similarity_score"])}')
    print('######')
    print(f'Mean LoRA: {np.mean(lora_df["lora_similarity_score"])}')
    print(f'Median LoRA: {np.median(lora_df["lora_similarity_score"])}')
    print(f'Stdev LoRA: {np.std(lora_df["lora_similarity_score"])}')

def print_KS_normality_results(score_df, lora_df):
    pixart_ks = kstest(score_df['pixart_similarity_score'], "norm")
    sdxl_ks = kstest(score_df['sdxl_similarity_score'], "norm")
    lora_sdxl_ks = kstest(lora_df['sdxl_similarity_score'], "norm")
    lora_ks = kstest(lora_df['lora_similarity_score'], "norm")
    print(f'Pixart Similarity Score Normality KS Test:{pixart_ks}')
    print(f'SDXL Similarity Score Normality KS Test:{sdxl_ks}')
    print(f'SDXL (Base, Monet) Similarity Score Normality KS Test:{lora_sdxl_ks}')
    print(f'LoRA (Monet) Similarity Score Normality KS Test:{lora_ks}')

def stat_comparison(df, column1, column2):
    # Two-sample Kolmogorov-Smirnov test
    ks_statistic, ks_p_value = ks_2samp(df[column1], df[column2])
    print("Two-sample Kolmogorov-Smirnov test:")
    print("KS statistic:", ks_statistic)
    print("P-value:", ks_p_value)

    #Wilcoxon Signed Rank test
    wilcoxon_statistic, wilcoxon_p_value = wilcoxon(df[column1], df[column2])
    print("Wilcoxon signed-rank test:")
    print("Wilcoxon statistic:", wilcoxon_statistic)
    print("P-value:", wilcoxon_p_value)

def main():
    score_df, lora_df = get_data()
    print_basic_stats(score_df, lora_df)
    print_KS_normality_results(score_df, lora_df)
    stat_comparison(score_df, 'sdxl_similarity_score', 'pixart_similarity_score')
    stat_comparison(lora_df, 'sdxl_similarity_score', 'lora_similarity_score')

if __name__ == "__main__":
    main()