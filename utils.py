import numpy as np
import pandas as pd
import os
from functools import lru_cache
from tqdm.auto import tqdm

DATA_FOLDER = "/mnt/md1/jjusuf/looping_review/simulation_data_processed"

DELTA_T = 0.02

def _load_mat_array(loop_num: int, rep: int, noise: int, non_sticky: bool=False) -> np.ndarray:
    """Load the numpy array for the given loop/rep/noise.
    Use the non_sticky argument to load the non-sticky simulations
    for EP loops."""

    if non_sticky:
        mat_path = os.path.join(
            DATA_FOLDER,
            f"loop_{loop_num}_non_sticky_{rep}_step_1_noise_{noise}um.txt",
        )
    else:
        mat_path = os.path.join(
            DATA_FOLDER,
            f"loop_{loop_num}_{rep}_step_1_noise_{noise}um.txt",
        )

    mat = np.loadtxt(mat_path)

    return mat

@lru_cache(maxsize=16)
def _load_trajectory_arrays(loop_num: int, rep: int, noise: int, non_sticky: bool=False) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the time points and 3D distances of a trajectory. Cached, because parsing
    the text file takes ~1 s; 16 trajectories take ~800 MB. The cached arrays are
    shared between calls, so callers must copy them before handing them out.
    """
    arr = _load_mat_array(loop_num, rep, noise, non_sticky=non_sticky)

    # time points always in first column
    time_index = arr[:, 0].copy()  # copy so the cache doesn't keep the whole matrix alive

    # 3D distance always in last column
    values = arr[:, -1].copy()

    return time_index, values

def load_trajectory(loop_num: int, rep: int, noise: int, non_sticky: bool=False) -> pd.Series:
    """
    Load the trajectory of 3D distance over time for a given loop, replicate,
    and amount of noise, as a time-indexed pandas Series. Use the non-sticky
    argument to load the non-sticky trajectories of EP loops.
    The 16 most recently loaded trajectories are cached; each call returns a new copy.
    """
    time_index, values = _load_trajectory_arrays(int(loop_num), int(rep), int(noise), bool(non_sticky))
    time_index, values = time_index.copy(), values.copy()  # callers may modify the Series in place

    if non_sticky:
        non_sticky_str = '_non_sticky'
    else:
        non_sticky_str = ''
    name = f"dist_3D_loop{loop_num}_rep{rep}_noise{noise}nm{non_sticky_str}"

    return pd.Series(values, index=time_index, name=name)

def dist_3D_stats_all_reps(
    loop_num: int,
    noise: int,
    non_sticky: bool = False,
    pbar: tqdm | None = None,
    reps: np.ndarray | None = None
) -> tuple[float, float]:
    """
    Get the mean 3D distance, median, mean error on 3D distance, and median error of a given loop
    when a certain amount of noise is added, using data from all reps.
    """

    if reps is None:
        reps = range(10, 19 + 1)  # 10..19 inclusive

    all_dist_3D = []
    diffs = []

    for rep in reps:
        traj_no_noise = load_trajectory(loop_num, rep, 0, non_sticky)
        traj_noise = load_trajectory(loop_num, rep, noise, non_sticky)
        diff = traj_noise - traj_no_noise  # the difference between measured vs true 3D distance
        all_dist_3D.append(traj_noise)
        diffs.append(diff)

        if pbar is not None:
            pbar.update(1)
    
    all_dist_3D = np.concatenate(all_dist_3D)
    diffs = np.concatenate(diffs)

    mean = np.mean(all_dist_3D)
    median = np.median(all_dist_3D)
    mean_err = np.mean(np.abs(diff))
    median_err = np.median(np.abs(diff))

    return mean, median, mean_err, median_err

def load_ctcf_state(loop_num: int, rep: int) -> pd.Series:
    """
    Load the binary CTCF state (0/1) over time for loops 0, 1, 2,
    as a time-indexed pandas Series.
    """
    if loop_num not in (0, 1, 2):
        raise ValueError("CTCF state is only defined for loops 0, 1, and 2")

    arr = _load_mat_array(loop_num, rep, noise=0)  # set noise to 0, as it doesn't matter

    # time points always in first column
    time_index = arr[:, 0]

    # For CTCF loops, binary state is stored in column 7
    values = arr[:, 7]

    name = f"ctcf_state_loop{loop_num}_rep{rep}"

    return pd.Series(values, index=time_index, name=name)

def event_lifetimes(
    state: pd.Series,
    ignore_changes_time: float,
    dt: float,
) -> pd.Series:
    """
    Compute event lifetimes from a binary (0/1) time series, using the
    same frame-based logic as the provided numpy implementation.

    Parameters
    ----------
    state : pd.Series
        Time-indexed binary Series (0 = unlooped, 1 = looped).
        Index should be regularly spaced in time.
    ignore_changes_time : float
        Threshold in seconds:
        - 0-gaps with length <= ignore_changes_time are bridged (set to 1).
        - Events with length < ignore_changes_time are discarded.
        Events touching either end of the series (after bridging) are
        incomplete and are always discarded.
    dt : float
        Time between frames in seconds

    Returns
    -------
    pd.Series
        Lifetimes of events (in time units of `state.index`),
        one value per event.
    """
    if len(state) < 2:
        return pd.Series(dtype=float, name="lifetime")

    # Binary track in frames
    binary_track = state.astype(int).to_numpy()

    # ---- First pass: find starts/ends ----
    # starts[i] is the first frame of event i, ends[i] is the frame after its last frame
    diffs = np.diff(np.pad(binary_track * 1, pad_width=(1, 1)))  # pad with zeros
    starts = np.where(diffs == 1)[0]   # transitions 0 -> 1
    ends = np.where(diffs == -1)[0]    # transitions 1 -> 0

    # ---- Bridge short gaps between events ----
    gaps = starts[1:] - ends[:-1]  # gap between end(i) and start(i+1)
    for i, gap in enumerate(gaps):
        if gap * dt <= ignore_changes_time:
            # Fill the gap between event i and event i+1
            binary_track[ends[i]:starts[i + 1]] = 1

    # ---- Recompute starts/ends after bridging ----
    diffs = np.diff(np.pad(binary_track * 1, pad_width=(1, 1)))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    # ---- Drop incomplete events (those touching the first or last frame) ----
    complete = (starts > 0) & (ends < len(binary_track))
    starts = starts[complete]
    ends = ends[complete]

    if len(starts) == 0:
        return pd.Series(dtype=float, name="lifetime")

    # Event lengths
    lifetimes_frames = ends - starts
    lifetimes_seconds = lifetimes_frames * dt

    # Drop any events with length < ignore_changes_time (in frames)
    mask = lifetimes_seconds >= ignore_changes_time
    lifetimes_seconds = lifetimes_seconds[mask]

    if lifetimes_seconds.size == 0:
        return pd.Series(dtype=float, name="lifetime")

    return pd.Series(lifetimes_seconds, name="lifetime").reset_index(drop=True)

def ctcf_event_lifetimes_all_reps(
    loop_num: int,
    ignore_changes_time: float = 0,
    target_frame_duration : float = 1,
    pbar: tqdm | None = None,
    reps: np.ndarray | None = None
) -> np.ndarray:
    """
    Get CTCF event lifetimes merged across replicates for a given loop.
    Note that all timepoints are always used; time sampling is not performed here.
    If `pbar` is provided, update it once per replicate.
    """
    sample_every = int(np.round(target_frame_duration / DELTA_T))
    actual_frame_duration = sample_every * DELTA_T

    lifetimes_all: list[pd.Series] = []

    if reps is None:
        reps = range(10, 19 + 1)  # 10..19 inclusive

    for rep in reps:
        ctcf_state = load_ctcf_state(loop_num, rep)
        lifetimes_this_rep = event_lifetimes(
            ctcf_state,
            ignore_changes_time = ignore_changes_time,
            dt = actual_frame_duration
        )
        lifetimes_all.append(lifetimes_this_rep)

        if pbar is not None:
            pbar.update(1)

    if not lifetimes_all:
        return np.array([], dtype=float)

    lifetimes = pd.concat(lifetimes_all)
    return lifetimes.to_numpy()

def under_threshold_times_all_reps(
    loop_num: int,
    noise: int,
    non_sticky: bool = False,
    threshold: float = 50,
    ignore_changes_time: float = 0,
    target_frame_duration : float = 1,
    pbar: tqdm | None = None,
    reps: np.ndarray | None = None
) -> np.ndarray:
    """
    Get the durations for which the 3D distance falls below a certain threshold,
    given the loop number and noise to load, sampling at the given frame duration.
    If `pbar` is provided, update it once per replicate.
    """
    
    sample_every = int(np.round(target_frame_duration / DELTA_T))
    actual_frame_duration = sample_every * DELTA_T
    
    lifetimes_all: list[pd.Series] = []

    if reps is None:
        reps = range(10, 19 + 1)  # 10..19 inclusive

    for rep in reps:        
        traj = load_trajectory(loop_num, rep, noise, non_sticky)
        
        # sample at desired rate
        traj = traj[::sample_every]
        
        lifetimes_this_rep = event_lifetimes(
            traj<threshold,
            ignore_changes_time = ignore_changes_time,
            dt = actual_frame_duration
        )
        lifetimes_all.append(lifetimes_this_rep)

        if pbar is not None:
            pbar.update(1)

    if not lifetimes_all:
        return np.array([], dtype=float)

    lifetimes = pd.concat(lifetimes_all)
    return lifetimes.to_numpy()

def apply_temporal_threshold(binary_arr: np.ndarray, temporal_threshold: int) -> np.ndarray:
    '''Given a binary array, filters out events that are shorter than a specified number of frames.'''
    filtered_arr = binary_arr.copy()
    padded_arr = np.r_[0, binary_arr, 0]
    interval_starts = np.where(np.diff(padded_arr) == 1)[0]
    interval_ends = np.where(np.diff(padded_arr) == -1)[0]

    for start, end in zip(interval_starts, interval_ends):
        if end - start < temporal_threshold:
            filtered_arr[start:end] = 0

    return filtered_arr

def apply_moving_average(binary_arr: np.ndarray, window_size: int) -> np.ndarray:
    '''
    Given a binary array, apply smoothing by using a centered moving average with the provided window size in frames.
    Frames beyond either end of the array count as 0. Uses a cumulative sum, so it takes O(n) time for any window size.

    An odd window (2m+1 frames) is centered on each frame i: frames i-m..i+m, each with weight 1.
    An even window (2m frames) cannot be centered on a frame, and shifting it half a frame either way moves every
    smoothed edge by one frame. Instead it is averaged over both shifts: frames i-m..i+m, with the two end frames
    weighted 1/2 (total weight 2m; the standard "2 x 2m" centered moving average).

    After apply_temporal_threshold with the same window size, thresholding the result at > 0.5 leaves the edges of
    every event unchanged and fills the gaps between events that are shorter than window_size / 2 frames.
    '''
    if window_size < 1:
        raise ValueError(f'window_size must be at least 1 frame, got {window_size}')

    n = len(binary_arr)
    half = window_size // 2
    # zero-pad so that frames i-half..i+half are padded[i..i+2*half]
    zeros = np.zeros(half, dtype=np.int64)
    padded = np.concatenate([zeros, np.asarray(binary_arr, dtype=np.int64), zeros])
    cumsum = np.concatenate([[0], np.cumsum(padded)])
    i = np.arange(n)
    window_sum = cumsum[i + 2 * half + 1] - cumsum[i]  # sum of frames i-half..i+half

    if window_size % 2 == 1:
        return window_sum / window_size

    # even window: the two end frames (i-half and i+half) only get weight 1/2
    end_frames = padded[i] + padded[i + 2 * half]
    return (window_sum - 0.5 * end_frames) / window_size

def under_threshold_times_spatiotemporal_method(
    loop_num: int,
    rep: int,
    noise: int,
    non_sticky: bool = False,
    target_frame_duration: float = 1,
    threshold: float = 50,
    temporal_threshold: float = 180,
) -> pd.Series:
    """
    Get the event lifetimes using the spatiotemporal filtering method. target_frame_duration and temporal_threshold are in seconds.
    Events below the distance threshold that last less than temporal_threshold are removed, then the remaining events
    are smoothed with a centered moving average over temporal_threshold (see apply_moving_average) and thresholded at > 0.5.
    """

    sample_every = int(np.round(target_frame_duration / DELTA_T))
    actual_frame_duration = sample_every * DELTA_T
    # events must last at least temporal_threshold, i.e. at least this many frames (and always at least 1 frame);
    # the 1e-9 keeps floating-point error from rounding an exact multiple of the frame duration up
    temporal_threshold_frames = max(1, int(np.ceil(temporal_threshold / actual_frame_duration - 1e-9)))

    traj = load_trajectory(loop_num, rep, noise, non_sticky)[::sample_every]

    binary_array = np.array((traj < threshold) * 1)
    binary_array_filtered = apply_temporal_threshold(binary_array, temporal_threshold_frames)
    binary_array_smoothed = apply_moving_average(binary_array_filtered, temporal_threshold_frames)
    binary_array_final = (binary_array_smoothed > 0.5) * 1
    binary_array_final = pd.Series(binary_array_final, index=traj.index)

    return event_lifetimes(binary_array_final, ignore_changes_time=0, dt=actual_frame_duration)

def under_threshold_times_spatiotemporal_method_all_reps(
    loop_num: int,
    noise: int,
    non_sticky: bool = False,
    target_frame_duration: float = 1,
    threshold: float = 50,
    temporal_threshold: float = 180,
    pbar: tqdm | None = None,
    reps: np.ndarray | None = None
) -> np.ndarray:
    """
    Get the event lifetimes using the spatiotemporal filtering method, for all reps of a loop.
    target_frame_duration and temporal_threshold are in seconds.
    If `pbar` is provided, update it once per replicate.
    """

    lifetimes_all: list[pd.Series] = []

    if reps is None:
        reps = range(10, 19 + 1)  # 10..19 inclusive

    for rep in reps:
        lifetimes_this_rep = under_threshold_times_spatiotemporal_method(
            loop_num,
            rep,
            noise,
            non_sticky,
            target_frame_duration,
            threshold,
            temporal_threshold
        )
        lifetimes_all.append(lifetimes_this_rep)

        if pbar is not None:
            pbar.update(1)

    if not lifetimes_all:
        return np.array([], dtype=float)

    lifetimes = pd.concat(lifetimes_all)
    return lifetimes.to_numpy()

def prob_loop_CTCF_all_reps(
    loop_num: int,
    pbar: tqdm | None = None,
    reps: np.ndarray | None = None
) -> float:
    """
    Get the ground-truth looping probability of a CTCF loop (based on
    loop extruder data from the simulation), across all replicates.
    """
    
    if reps is None:
        reps = range(10, 19 + 1)  # 10..19 inclusive
    
    ctcf_states_all: list[np.ndarray] = []

    for rep in reps:        
        ctcf_state_this_rep = load_ctcf_state(loop_num, rep)
        ctcf_states_all.append(ctcf_state_this_rep.to_numpy())
           
        if pbar is not None:
            pbar.update(1)
            
    ctcf_states = np.concatenate(ctcf_states_all)
    looping_prob = np.mean(ctcf_states)
    
    return looping_prob

def prob_under_threshold_all_reps(
    loop_num: int,
    noise: int,
    non_sticky: bool = False,
    threshold: float = 50,
    pbar: tqdm | None = None,
    reps: np.ndarray | None = None
) -> float:
    """
    Get the probability that the 3D distance is under a given threshold,
    across all replicates.
    """

    if reps is None:
        reps = range(10, 19 + 1)  # 10..19 inclusive
    
    under_threshold_all: list[np.ndarray] = []

    for rep in reps:        
        under_threshold_this_rep = (load_trajectory(loop_num, rep, noise, non_sticky).to_numpy() < threshold)
        under_threshold_all.append(under_threshold_this_rep)
           
        if pbar is not None:
            pbar.update(1)
            
    under_threshold = np.concatenate(under_threshold_all)
    under_threshold_prob = np.mean(under_threshold)
    
    return under_threshold_prob

def dist_3D_distribution_all_reps(
    loop_num: int,
    noise: int,
    non_sticky: bool = False,
    pbar: tqdm | None = None,
    reps: np.ndarray | None = None
) -> np.ndarray:
    """
    Get an array containing all 3D distances at all timepoints across,
    all replicates of a given loop. In reality, this array is just a
    concatenation of all the 3D distance trajectories across replicates.
    """

    if reps is None:
        reps = range(10, 19 + 1)  # 10..19 inclusive
    
    dist_3D_all: list[np.ndarray] = []

    for rep in reps:        
        dist_3D_this_rep = load_trajectory(loop_num, rep, noise, non_sticky).to_numpy()
        dist_3D_all.append(dist_3D_this_rep)
           
        if pbar is not None:
            pbar.update(1)
            
    dist_3D_distribution = np.concatenate(dist_3D_all)
    
    return dist_3D_distribution

def search_times(
    loop_num: int,
    rep: int,
    noise: int,
    non_sticky: bool = False,
    threshold: float = 50,
    target_frame_duration: float = 1,
    median: float | None = None,
    samples_per_event: int = 1
) -> np.ndarray:
    """
    Get an array of search times for a given loop and replicate.
    If median is not provided, it will be calculated here.
    """
    
    sample_every = int(np.round(target_frame_duration / DELTA_T))
    actual_frame_duration = sample_every * DELTA_T
    
    traj = load_trajectory(loop_num, rep, noise, non_sticky)[::sample_every]
    
    if median is None:
        median = np.median(traj)
        
    proximal_state = 1*np.logical_and(traj>=0, traj<threshold)
    distal_state = 1*np.logical_and(traj>=median, traj<median+threshold)
    
    distal_state_indices = np.where(distal_state==1)[0]
    
    proximal_state_starts = np.where(np.concatenate((np.diff(proximal_state), [0]))==1)[0]  # indices where proximal state began
    proximal_state_ends = np.where(np.concatenate((np.diff(proximal_state), [0]))==-1)[0]  # indices where proximal state ended

    proximal_state_starts = proximal_state_starts[1:]  # remove first event
    proximal_state_ends = proximal_state_ends[:-1]  # for each matching index, this will contain the previous end
    
    # now get the periods leading up to each proximal state start
    num_periods = min(len(proximal_state_starts), len(proximal_state_ends))
    non_proximal_periods = [np.arange(proximal_state_ends[k], proximal_state_starts[k])
                            for k in range(num_periods)]

    search_time_arr = []
    
    for i, period in enumerate(non_proximal_periods):
        proximal_state_start = proximal_state_starts[i]
        distal_indices_to_select = np.intersect1d(period, distal_state_indices)  # which distal indices to choose from
        if len(distal_indices_to_select)>0:
            for s in range(samples_per_event):
                distal_index_chosen = np.random.choice(distal_indices_to_select)
                search_time = (proximal_state_start-distal_index_chosen) * actual_frame_duration
                search_time_arr.append(search_time)
        else:
            continue  # it wasn't in the distal state at all
    
    return np.array(search_time_arr)

def search_times_CTCF(
    loop_num: int,
    rep: int,
    noise: int,
    non_sticky: bool = False,
    threshold: float = 50,
    target_frame_duration: float = 1,
    median: float | None = None,
    samples_per_event: int = 1,
) -> np.ndarray:
    """
    Get an array of search times for a given loop and replicate.
    For CTCF, the proximal state is defined as a true CTCF looping event.
    If median is not provided, it will be calculated here.
    """
    
    sample_every = int(np.round(target_frame_duration / DELTA_T))
    actual_frame_duration = sample_every * DELTA_T
    
    traj = load_trajectory(loop_num, rep, noise, non_sticky)[::sample_every]
    
    if median is None:
        median = np.median(traj)
        
    proximal_state = load_ctcf_state(loop_num, rep)[::sample_every]
    distal_state = 1*np.logical_and(traj>=median, traj<median+threshold)
    
    distal_state_indices = np.where(distal_state==1)[0]
    
    proximal_state_starts = np.where(np.concatenate((np.diff(proximal_state), [0]))==1)[0]  # indices where proximal state began
    proximal_state_ends = np.where(np.concatenate((np.diff(proximal_state), [0]))==-1)[0]  # indices where proximal state ended

    proximal_state_starts = proximal_state_starts[1:]  # remove first event
    proximal_state_ends = proximal_state_ends[:-1]  # for each matching index, this will contain the previous end
    
    # now get the periods leading up to each proximal state start
    num_periods = min(len(proximal_state_starts), len(proximal_state_ends))
    non_proximal_periods = [np.arange(proximal_state_ends[k], proximal_state_starts[k])
                            for k in range(num_periods)]

    search_time_arr = []
    
    for i, period in enumerate(non_proximal_periods):
        proximal_state_start = proximal_state_starts[i]
        distal_indices_to_select = np.intersect1d(period, distal_state_indices)  # which distal indices to choose from
        if len(distal_indices_to_select)>0:
            for s in range(samples_per_event):
                distal_index_chosen = np.random.choice(distal_indices_to_select)
                search_time = (proximal_state_start-distal_index_chosen) * actual_frame_duration
                search_time_arr.append(search_time)
        else:
            continue  # it wasn't in the distal state at all
    
    return np.array(search_time_arr)

def search_times_all_reps(
    loop_num: int,
    noise: int,
    non_sticky: bool = False,
    threshold: float = 50,
    target_frame_duration: float = 1,
    pbar: tqdm | None = None,
    reps: np.ndarray | None = None,
    samples_per_event: int = 1
) -> np.ndarray:
    """
    Get an array of search times for a given loop, across all replicates.
    If median is not provided, it will be calculated here.
    """
    
    if reps is None:
        reps = range(10, 19 + 1)  # 10..19 inclusive
    
    # get the most accurate median first (across all reps)
    dist_3D_distribution = dist_3D_distribution_all_reps(loop_num, noise, non_sticky, reps=reps)
    median = np.median(dist_3D_distribution)
        
    search_times_all: list[np.ndarray] = []
        
    for rep in reps:
        search_times_this_rep = search_times(loop_num, rep, noise, non_sticky, threshold, target_frame_duration, median, samples_per_event)
        search_times_all.append(search_times_this_rep)
           
        if pbar is not None:
            pbar.update(1)
            
    search_times_arr = np.concatenate(search_times_all)
    
    return search_times_arr

def search_times_CTCF_all_reps(
    loop_num: int,
    noise: int,
    non_sticky: bool = False,
    threshold: float = 50,
    target_frame_duration: float = 1,
    pbar: tqdm | None = None,
    reps: np.ndarray | None = None,
    samples_per_event: int = 1
) -> np.ndarray:
    """
    Get an array of search times for a given loop, across all replicates.
    For CTCF, the proximal state is defined as a true CTCF looping event.
    If median is not provided, it will be calculated here.
    """
    
    if reps is None:
        reps = range(10, 19 + 1)  # 10..19 inclusive
    
    # get the most accurate median first (across all reps)
    dist_3D_distribution = dist_3D_distribution_all_reps(loop_num, noise, non_sticky, reps=reps)
    median = np.median(dist_3D_distribution)
        
    search_times_all: list[np.ndarray] = []
        
    for rep in reps:
        search_times_this_rep = search_times_CTCF(loop_num, rep, noise, non_sticky, threshold, target_frame_duration, median, samples_per_event)
        search_times_all.append(search_times_this_rep)
           
        if pbar is not None:
            pbar.update(1)
            
    search_times_arr = np.concatenate(search_times_all)
    
    return search_times_arr
