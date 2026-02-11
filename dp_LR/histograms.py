'''
Generate histograms for each dataset, i.e. orl faces and random.
'''

import matplotlib.pyplot as plt
import numpy as np
import argparse
from utils import load_orl_faces, generate_data, normalize_vector


def main():
    plt.rcParams.update({'font.size': 20})

    # Setup parser
    parser = argparse.ArgumentParser()

    parser.add_argument("--synthetic_data", action='store_true',
                        help="Plot histograms for the synthetic - random dataset. Default is the ORL faces dataset.")
    args = parser.parse_args()

    
    if args.synthetic_data:
        M, N, d = 10, 10, 100
        X_synth, y_synth, means_synth = generate_data(M=M, N=N, d=d, sigma=3, means="random")
        # Normalize
        for i, mean in enumerate(means_synth):
            X = normalize_vector(X_synth[i])
            means_synth[i] = normalize_vector(mean)
        for i, mean in enumerate(means_synth):
            plt.clf()
            plt.hist(mean, bins=30, color='skyblue', edgecolor='black')
            plt.xlabel('Values')
            plt.ylabel('Frequency')
            plt.title('Histogram of the vectorized mean\n of the image of ORL dataset')
            print(f"Saving fig results/noisy_weights/histograms/synthetic_label_{i}.png")
            plt.savefig(f'results/noisy_weights/histograms/synthetic_label_{i}.png')

    else:
        X, y, means = load_orl_faces('data/orl')
        for i, mean in enumerate(means):
            plt.clf()
            plt.hist(mean, bins=30, color='skyblue', edgecolor='black')
            plt.xlabel('Values')
            plt.ylabel('Frequency')
            plt.title('Histogram of the vectorized mean\n of the image of ORL dataset')
            print(f"Saving fig results/noisy_weights/histograms/orl_label{i}.png")
            plt.savefig(f'results/noisy_weights/histograms/orl_label{i}.png')



if __name__ == '__main__':
    main()
