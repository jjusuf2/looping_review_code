import numpy as np

import sys
from pathlib import Path
sys.path.append('../')
from utils import *  # key functions for this project

noise_arr = [10, 20, 30, 40, 50]

print(f'sigma_x   mean   median   mean_err   median_err')
for noise in noise_arr:
    mean, median, mean_err, median_err = dist_3D_stats_all_reps(loop_num=0, noise=noise, non_sticky=False)
    print(f'{noise} nm   {mean:.0f} nm   {median:.0f} nm   {mean_err:.0f} nm   {median_err:.0f} nm')
