import random

def main(save_path):
    random_seeds = [random.randint(0, 1000000) for _ in range(100)]
    with open(f'{save_path}random_seeds.txt', 'w') as file:
        for seed in random_seeds:
            file.write(str(seed) + '\n')

if __name__ == "__main__":
    main()