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

dt_arr = np.array([0.04])
loop_num_arr = np.array([3, 4, 5])
noise_arr = np.array([0, 10, 20, 30, 40, 50])

n_reps = 19 - 10 + 1  # 10..19 inclusive
total_iters = len(dt_arr) * len(loop_num_arr) * len(noise_arr) * n_reps

rows = []

with tqdm(total=total_iters, desc="calculating times below threshold", unit="rep") as pbar:
    for dt in dt_arr:
        for loop_num in loop_num_arr:
            for noise in noise_arr:
                under_threshold_times = under_threshold_times_all_reps(
                    loop_num=loop_num,
                    noise=noise,
                    threshold=THRESHOLD,
                    ignore_changes_time=dt,
                    target_frame_duration=dt,
                    pbar=pbar
                )
                for under_threshold_time in under_threshold_times:
                    rows.append(
                        {
                            "loop": loop_num,
                            "noise": noise,
                            "dt": dt,
                            "time": under_threshold_time,
                        }
                    )

under_threshold_times_df_EP = pd.DataFrame(rows)

under_threshold_times_df_EP.to_csv(f'../data/EP_loops_under_threshold_times_{THRESHOLD}nm_004.csv', index=False, float_format="%.2f")
