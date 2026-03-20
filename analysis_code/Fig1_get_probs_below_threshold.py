import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import sys
from pathlib import Path
sys.path.append('../')
from utils import *  # key functions for this project

THRESHOLD = 50

loop_nums = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
noise_levels = np.array([0, 10, 20, 30, 40, 50])
n_reps = 19 - 10 + 1  # 10..19 inclusive

probs_under_threshold = pd.DataFrame(index=loop_nums, columns=noise_levels, dtype='float')
probs_under_threshold.index.name = 'loop_num'
probs_under_threshold.columns.name = 'noise'
total_iters = len(loop_nums) * len(noise_levels) * n_reps

with tqdm(total=total_iters, desc="probability under threshold", unit="rep") as pbar:
    for loop_num in loop_nums:
        for noise in noise_levels:
            probs_under_threshold.loc[loop_num, noise] = prob_under_threshold_all_reps(loop_num, noise, non_sticky=False, threshold=THRESHOLD, pbar=pbar)
        
with open('../data/probs_under_threshold.csv', mode='w') as f:
    f.write('# Note: columns correspond to different noise levels (provided as sigma_x in nm)\n')
    probs_under_threshold.to_csv(f, float_format="%.6f")
