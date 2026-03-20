import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import pickle

import sys
from pathlib import Path
sys.path.append('../')
from utils import *  # key functions for this project

THRESHOLD = 50

dt_arr = np.array([0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 30, 60, 120])
loop_num_arr = np.array([3, 4, 5])
noise_arr = np.array([0, 10, 20, 30, 40, 50])
reps_arr = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])

for dt in dt_arr:
    print(f'working on dt={dt}', flush=True)
    if dt < 1:
        reps = [10]
    else:
        reps = reps_arr
    total_iters = len(loop_num_arr) * len(noise_arr) * len(reps)
    with tqdm(total=total_iters, desc="calculating search times", unit="rep") as pbar:
        rows_EP = []
        for loop_num in loop_num_arr:
            for noise in noise_arr:
                search_times_arr = search_times_all_reps(loop_num, noise, non_sticky=False, threshold=THRESHOLD,
                                                         target_frame_duration=dt, pbar=pbar, reps=reps)
                for t_search in search_times_arr:
                    rows_EP.append(
                        {
                            "loop": loop_num,
                            "noise": noise,
                            "time": t_search,
                        }
                    )

    search_times_EP = pd.DataFrame(rows_EP)
    search_times_EP.to_csv(f'../data/search_times_EP_deltaT_{dt}s.csv', index=False, float_format="%.2f")
