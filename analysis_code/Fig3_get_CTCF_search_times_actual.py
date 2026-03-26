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

dt_arr = 0.2
loop_num_arr = np.array([0, 1, 2])
noise = 0
n_reps = 10

for dt in dt_arr:
    print(f'working on dt={dt}', flush=True)
    total_iters = len(loop_num_arr) * n_reps
    with tqdm(total=total_iters, desc="calculating search times", unit="rep") as pbar:
        rows_CTCF = []
        for loop_num in loop_num_arr:
            search_times_arr = search_times_CTCF_all_reps(loop_num, noise, non_sticky=False, threshold=THRESHOLD,
                                                            target_frame_duration=dt, pbar=pbar)
            for t_search in search_times_arr:
                rows_CTCF.append(
                    {
                        "loop": loop_num,
                        "time": t_search,
                    }
                )

    search_times_CTCF = pd.DataFrame(rows_CTCF)
    search_times_CTCF.to_csv(f'../data/search_times_CTCF_actual.csv', index=False, float_format="%.2f")
