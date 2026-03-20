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

loop_num_arr = np.array([0, 1, 2])

n_reps = 19 - 10 + 1  # 10..19 inclusive
total_iters = len(loop_num_arr) * n_reps

rows = []

with tqdm(total=total_iters, desc="calculating CTCF lifetimes", unit="rep") as pbar:
    for loop_num in loop_num_arr:
        lifetimes = ctcf_event_lifetimes_all_reps(
            loop_num=loop_num,
            ignore_changes_time=30,
            pbar=pbar
        )
        for life in lifetimes:
            rows.append(
                {
                    "loop": loop_num,
                    "lifetime": life,
                }
            )

lifetimes_df = pd.DataFrame(rows)

lifetimes_df.to_csv(f'../data/CTCF_loops_lifetimes.csv', index=False, float_format="%.2f")