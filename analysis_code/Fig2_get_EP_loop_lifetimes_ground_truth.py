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

# get these tables from earlier
t_under_50nm_EP = pd.read_csv(f'../data/EP_loops_under_threshold_times_50nm.csv')
t_under_50nm_EP_ns = pd.read_csv(f'../data/EP_loops_non_sticky_under_threshold_times_50nm.csv')

loop_num_arr = np.array([3, 4, 5])

EP_loops_lifetime_histograms = pd.DataFrame(index=loop_num_arr, columns=['bin_edges', 'hist', 'hist_non_sticky', 'hist_diff', 'mean'])

for loop_num in loop_num_arr:
    lifetimes = t_under_50nm_EP.loc[(t_under_50nm_EP['loop']==loop_num) & (t_under_50nm_EP['noise']==0) & (t_under_50nm_EP['dt']==0.1), 'time']
    lifetimes_ns = t_under_50nm_EP_ns.loc[(t_under_50nm_EP_ns['loop']==loop_num) & (t_under_50nm_EP_ns['noise']==0) & (t_under_50nm_EP_ns['dt']==0.1), 'time']
    bin_size = 0.2
    hist, bin_edges = np.histogram(lifetimes, bins=np.arange(0, 10, bin_size))
    hist_ns, _ = np.histogram(lifetimes_ns, bins=np.arange(0, 10, bin_size))
    hist_diff = hist-hist_ns
    hist_diff[hist_diff<0] = 0
    bin_centers = (bin_edges[1:]+bin_edges[:-1])/2
    mean = np.sum(hist_diff*bin_centers)/np.sum(hist_diff)

    # save results
    EP_loops_lifetime_histograms.loc[loop_num] = bin_edges, hist, hist_ns, hist_diff, mean

EP_loops_lifetime_histograms.to_pickle('../data/EP_loops_lifetime_histograms.pkl')