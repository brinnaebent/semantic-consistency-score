import random

def main(savepath):
    # Define the model order
    models_order_list = [["SDXL", "pixart"] if random.random() < 0.5 else ["pixart", "SDXL"] for _ in range(len(common_folder_names))]

    # Save the model order to a file
    with open(f'{savepath}user_research_model_order.txt', 'w') as f:
        for order_pair in models_order_list:
            f.write(f"{order_pair[0]},{order_pair[1]}\n")

if __name__ == "__main__":
    main()