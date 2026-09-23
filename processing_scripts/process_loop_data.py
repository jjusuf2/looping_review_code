import numpy as np
import pandas as pd
from polychrom.hdf5_format import list_URIs, load_URI
from tqdm import tqdm
import sys

loop_name = sys.argv[1]
block_start = int(sys.argv[2]) # 1000000
block_end = int(sys.argv[3]) # 4000000
block_step = int(sys.argv[4]) # 50

sim_calib_curve_data_subset = pd.read_csv('/mnt/md0/jjusuf/looping_review/loop_info.csv', index_col=0)

files = list_URIs('/mnt/md1/jjusuf/looping_review/sim_data/sim_20260115_with_LE_with_EP_main_save_every_block/')

chrom_size = 70000
region_size = 2000
num_regions = chrom_size // region_size
region_starts = np.arange(num_regions) * region_size

CTCF_sites_L = np.array([574, 694, 866, 1241, 1390, 1580, 1752, 1800])
CTCF_sites_R = np.array([200, 330, 724, 1425, 1433, 1604])
sticky_elements = np.array([250, 372, 540, 745, 775, 833, 961, 1202, 1330, 1640, 1722])

random_elements = np.array([102, 421, 450, 499, 1050, 1902])
sites_of_interest_in_region = np.concatenate((CTCF_sites_L, CTCF_sites_R, sticky_elements, random_elements))
sites_of_interest = np.repeat(region_starts,len(sites_of_interest_in_region)) + np.tile(sites_of_interest_in_region, num_regions)

lookup_pd = pd.Series(np.arange(len(sites_of_interest)), index=sites_of_interest)
lookup = dict(lookup_pd)

class UnionFind:
    def __init__(self):
        self.parent = {}
    def find(self, x):
        if x != self.parent.setdefault(x, x):
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, x, y):
        self.parent[self.find(x)] = self.find(y)
    def connected(self, x, y):
        return self.find(x) == self.find(y)

def in_contact(pos1, pos2, SMC_pos):
    if pos1 in SMC_pos and pos2 in SMC_pos:
        unique_points = np.unique(SMC_pos)
        SMC_pos_ext = np.concatenate((SMC_pos, np.array([unique_points, unique_points+1]).T, np.array([unique_points, unique_points-1]).T), axis=0)
        uf = UnionFind()
        for x, y in SMC_pos_ext:
            uf.union(x, y)
        return uf.connected(pos1, pos2)
    else:
        return False

CTCF_loop_names = ['loop_0','loop_1','loop_2']

repeat_nums = list(range(10, 20))  # replicates 10–19

# ---- Precompute static left/right offsets ----
base_left  = sim_calib_curve_data_subset.loc[loop_name, 'left']
base_right = sim_calib_curve_data_subset.loc[loop_name, 'right']

# Precompute left/right monomer IDs for each replicate
left_ids  = [r*2000 + base_left  for r in repeat_nums]
right_ids = [r*2000 + base_right for r in repeat_nums]

# Precompute lookup indices (do once!)
left_indices  = [lookup[l] for l in left_ids]
right_indices = [lookup[r] for r in right_ids]

# ---- Open all output files once ----
outfiles = []
for r in repeat_nums:
    fname = f"/mnt/md1/jjusuf/looping_review/simulation_data_processed/{loop_name}_{r}_step_{block_step}.txt"
    outfiles.append(open(fname, "w"))

# ---- Main streaming loop ----
for block_num in tqdm(range(block_start, block_end, block_step)):

    data = load_URI(files[block_num])   # EXPENSIVE — now done once
    pos_data = data['pos']
    SMC_data = data['SMCs']

    for i, r in enumerate(repeat_nums):

        left_index  = left_indices[i]
        right_index = right_indices[i]

        left_vals  = pos_data[left_index, :]
        right_vals = pos_data[right_index, :]

        # Fast formatting (avoid np.char.mod)
        line = f"{block_num}\t{left_vals[0]:.2f}\t{left_vals[1]:.2f}\t{left_vals[2]:.2f}\t{right_vals[0]:.2f}\t{right_vals[1]:.2f}\t{right_vals[2]:.2f}"

        if loop_name in CTCF_loop_names:
            contact = int(in_contact(left_ids[i], right_ids[i], SMC_data))
            line += f"\t{contact}"

        outfiles[i].write(line + "\n")

# ---- Close files ----
for f in outfiles:
    f.close()
