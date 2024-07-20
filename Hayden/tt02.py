import csv
import datetime, neo
import json
import os
import re
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from scipy.signal import filtfilt, butter, firwin, lfilter, medfilt
from scipy.spatial.distance import mahalanobis
from scipy.linalg import inv
import matplotlib.pyplot as plt
from pathlib import Path
from collections import namedtuple
from scipy import ndimage
from scipy.ndimage import gaussian_filter, convolve
from datetime import datetime
import glob
from scipy.stats import chi2
from scipy.stats import multivariate_normal
from scipy.ndimage import label
from skimage.feature import peak_local_max
from scipy.interpolate import interp1d

# Define parameter settings
PARAM = {
    'theta_band': [6, 10],
    'decim_r': 8,
}


def load_targets(filename):
    with open(filename, 'r') as file:
        return file.readlines()


def map_behavior(coords, *args):
    """
    Map a set of coordinates to a grid.
    """
    if len(coords) == 0:
        return

    options = {'filters': [], 'grid': []}
    options, other, remainder = parse_args(args, options)

    if not other:
        vars = []
    else:
        vars = other[0]

    if isinstance(coords, np.ndarray):
        coords = [coords]
    elif not isinstance(coords, list):
        raise ValueError("Invalid coordinates")

    N = len(coords)

    if not vars:
        vars = [None] * N
    elif isinstance(vars, np.ndarray) and N == 1:
        vars = [vars]
    elif not isinstance(vars, list) or len(vars) != N:
        raise ValueError("Invalid variables")

    m = []

    for k in range(len(coords)):
        if options['filters']:
            coords[k] = filtercoords(coords[k], options['filters'])

        bins, options['grid'] = coords_to_bin(coords[k], options['grid'])
        sz = options['grid'].shape

        m.append(bin_to_map(bins, vars[k], *remainder, size=sz))

    m = np.concatenate(m, axis=len(options['grid'].shape))
    g = options['grid']
    return m, g

def filtercoords(coords, filters):
    """
    Apply a set of filters to the coordinates.
    """
    filtered_coords = coords
    for f in filters:
        filtered_coords = f(filtered_coords)
    return filtered_coords

def coords_to_bin(coords, grid=None):
    """
    Map coordinates to a grid.
    """
    if len(coords) == 0:
        return np.array([]), grid

    coords = np.asarray(coords)
    n, m = coords.shape
    maxcoords = coords.max(axis=0)
    mincoords = coords.min(axis=0)

    edges = []

    if grid is None:
        grid = 10

    if isinstance(grid, (int, float)) and np.isscalar(grid):
        grid = np.full(m, grid)

    if isinstance(grid, np.ndarray) and grid.ndim == 1 and len(grid) == m:
        for k in range(m):
            if maxcoords[k] == mincoords[k]:
                edges.append(np.linspace(maxcoords[k] - 0.5 * grid[k], maxcoords[k] + 0.5 * grid[k], int(grid[k])))
            else:
                edges.append(np.linspace(mincoords[k], maxcoords[k], int(grid[k]) + 1))
        grid = np.array(edges)

    b = binning(grid, coords)
    return b, grid

def parse_args(args, argstruct):
    """
    Helper function for parsing varargin.
    """
    remainder = []
    otherargs = []

    if not args:
        return argstruct, otherargs, remainder

    valid_args = list(argstruct.keys())
    valid_args_size = [len(arg) for arg in valid_args]

    num_other = next((i for i, x in enumerate(args) if isinstance(x, str)), len(args))
    if num_other > 0:
        otherargs = args[:num_other]

    for k in range(num_other, len(args), 2):
        if not isinstance(args[k], str):
            raise ValueError("Expected a named argument")

        matches = [i for i, arg in enumerate(valid_args) if arg.startswith(args[k])]

        if not matches:
            if remainder is not None:
                remainder.extend(args[k:k+2])
            else:
                raise ValueError(f"Unknown named parameter: {args[k]}")
        elif len(matches) > 1:
            midx = min((valid_args_size[i], i) for i in matches)[1]
            argstruct[valid_args[midx]] = args[k + 1]
        else:
            argstruct[valid_args[matches[0]]] = args[k + 1]

    return argstruct, otherargs, remainder

def binning(grid, coords):
    """
    Bin coordinates using a grid.
    """
    coords = np.asarray(coords)
    b = np.zeros_like(coords, dtype=int)

    for k in range(grid.shape[0]):
        if grid[k].ndim == 1:
            b[:, k] = np.digitize(coords[:, k], grid[k]) - 1
            b[b[:, k] == len(grid[k]), k] = 0

    return b

def bin_to_map(bins, vars=None, **kwargs):
    """
    Create histogram of variables at binned coordinates.
    """
    options = {'default': np.nan, 'function': np.size, 'size': None}
    options.update(kwargs)

    bins = np.asarray(bins)
    n, m = bins.shape

    if vars is None:
        vars = np.ones((n, 1))
    else:
        vars = np.asarray(vars)
        if vars.shape[0] != n:
            raise ValueError("Invalid variables")

    var_ndims = vars.ndim
    var_size = vars.shape

    if options['size'] is None:
        map_size = bins.max(axis=0)
    else:
        map_size = np.array(options['size'])

    if map_size.size != m:
        raise ValueError("Invalid size option")

    mp = np.full(map_size.tolist() + list(var_size[1:]), options['default'])

    valids = (bins != 0).all(axis=1)
    if not valids.any():
        return mp

    unique_bins, indices = np.unique(bins[valids], axis=0, return_inverse=True)
    rb, cb = unique_bins.shape

    for d in range(rb):
        mask = indices == d
        mp[tuple(unique_bins[d])] = options['function'](vars[valids][mask], axis=0)

    return mp

def load_target_flist(path_to_trg_dir, w_card):
    if not isinstance(path_to_trg_dir, (str, list)):
        raise ValueError('load_target_flist: path_to_trg_dir must be a string or list')
    if not isinstance(w_card, (str, list)):
        raise ValueError('load_target_flist: w_card must be a string or list')

    if isinstance(path_to_trg_dir, str):
        path_to_trg_dir = [path_to_trg_dir]
    if isinstance(w_card, str):
        w_card = [w_card]

    if len(path_to_trg_dir) != len(w_card):
        raise ValueError('load_target_flist: Length of path_to_trg_dir must be equal to length of w_card')

    list_targets = [None] * len(path_to_trg_dir)

    for i, (trg_dir, trg_wcard) in enumerate(zip(path_to_trg_dir, w_card)):
        if not os.path.isdir(trg_dir):
            raise ValueError('load_target_flist: path_to_trg_dir must be a list of valid paths to target folders')

        trg_list = glob.glob(os.path.join(trg_dir, trg_wcard))
        if not trg_list:
            raise ValueError('load_target_flist: No target files found or error in w_card syntax')

        trg_list = [os.path.abspath(name) for name in trg_list]

        if len(path_to_trg_dir) == 1:
            list_targets = trg_list
        else:
            list_targets[i] = trg_list

    return list_targets


def eegfilt(data, srate, locutoff, hicutoff, epochframes=0, filtorder=0, revfilt=False, firtype='fir1', causal=False):
    if data.ndim == 1:
        data = data.reshape(1, -1)  # Convert to 2D array

    chans, frames = data.shape
    nyq = srate * 0.5  # Nyquist frequency
    minfreq = 0  # Minimum frequency
    minfac = 3  # Minimum factor for cutoff frequency cycles in filter
    min_filtorder = 15  # Minimum filter length
    trans = 0.15  # Fractional width of transition zones

    if locutoff > 0 and hicutoff > 0 and locutoff > hicutoff:
        raise ValueError('locutoff > hicutoff ???')
    if locutoff < 0 or hicutoff < 0:
        raise ValueError('locutoff or hicutoff < 0 ???')
    if locutoff > nyq:
        raise ValueError('Low cutoff frequency cannot be > srate/2')
    if hicutoff > nyq:
        raise ValueError('High cutoff frequency cannot be > srate/2')

    if filtorder == 0:
        if locutoff > 0:
            filtorder = minfac * int(srate / locutoff)
        elif hicutoff > 0:
            filtorder = minfac * int(srate / hicutoff)
        if filtorder < min_filtorder:
            filtorder = min_filtorder

    if epochframes == 0:
        epochframes = frames
    epochs = frames // epochframes
    if epochs * epochframes != frames:
        raise ValueError('epochframes does not divide frames.')

    if filtorder * 3 > epochframes:
        raise ValueError(f'epochframes must be at least 3 times the filtorder. Current filtorder is {filtorder}.')
    if (1 + trans) * hicutoff / nyq > 1:
        raise ValueError('high cutoff frequency too close to Nyquist frequency')

    if locutoff > 0 and hicutoff > 0:
        if firtype == 'fir1':
            filtwts = firwin(filtorder, [locutoff, hicutoff], pass_zero=False, nyq=nyq)
    elif locutoff > 0:
        if locutoff / nyq < minfreq:
            raise ValueError(f'highpass cutoff freq must be > {minfreq * nyq} Hz')
        if firtype == 'fir1':
            filtwts = firwin(filtorder, locutoff, pass_zero=False, nyq=nyq)
    elif hicutoff > 0:
        if hicutoff / nyq < minfreq:
            raise ValueError(f'lowpass cutoff freq must be > {minfreq * nyq} Hz')
        if firtype == 'fir1':
            filtwts = firwin(filtorder, hicutoff, nyq=nyq)

    if revfilt:
        filtwts = -filtwts

    smoothdata = np.zeros((chans, frames))
    for e in range(epochs):
        for c in range(chans):
            epoch_data = data[c, e * epochframes:(e + 1) * epochframes]
            if causal:
                smoothdata[c, e * epochframes:(e + 1) * epochframes] = lfilter(filtwts, 1, epoch_data)
            else:
                smoothdata[c, e * epochframes:(e + 1) * epochframes] = filtfilt(filtwts, 1, epoch_data)

    return smoothdata.T


def spk_select_fet(fet_in, cids_in, do_remove_cluster_zero=False):
    if fet_in.ndim != 2:
        raise ValueError('spk_select_fet: fet_in must be a 2D matrix')

    if not isinstance(cids_in, (list, np.ndarray)):
        raise ValueError('spk_select_fet: cids_in must be a vector')

    if fet_in.shape[1] != len(cids_in):
        raise ValueError('spk_select_fet: fet_in and cids_in must be equal length')

    cids_unique = np.unique(cids_in)  # unique cluster IDs

    if len(cids_unique) == 1 and cids_unique[0] == 0:
        raise ValueError('spk_select_fet: unsorted spikes provided')

    # remove cluster zero (unsorted spikes)
    if do_remove_cluster_zero:
        if cids_unique[0] == 0:
            cids_unique = cids_unique[1:]

    ncells = len(cids_unique)  # total number of cells (clusters) in the file

    # prepare storage for output
    fet = [None] * ncells
    cids = [None] * ncells

    # select per-cell spike features from all features
    for ci in range(ncells):
        fet[ci] = fet_in[:, cids_in == cids_unique[ci]]
        cids[ci] = cids_unique[ci]

    return fet, cids


def spk_select_ts(ts_in, cids_in, do_remove_cluster_zero=False):
    """
    SpkSelectTs provides follow up processing of spike timestamps extracted
    from sorted ntt/nst/nse files by using NlxGetSpikes* set of functions.
    """
    if not isinstance(ts_in, (list, np.ndarray)):
        raise ValueError('spk_select_ts: ts_in must be a vector')

    if not isinstance(cids_in, (list, np.ndarray)):
        raise ValueError('spk_select_ts: cids_in must be a vector')

    if len(ts_in) != len(cids_in):
        raise ValueError('spk_select_ts: ts_in and cids_in must be equal length')

    cids_unique = np.unique(cids_in)  # unique cluster IDs

    if len(cids_unique) == 1 and cids_unique[0] == 0:
        raise ValueError('spk_select_ts: unsorted spikes provided')

    # remove cluster zero (unsorted spikes)
    if do_remove_cluster_zero:
        if cids_unique[0] == 0:
            cids_unique = cids_unique[1:]

    ncells = len(cids_unique)  # total number of cells (clusters) in the file

    # prepare storage for output
    ts = [None] * ncells
    cids = [None] * ncells

    # select per-cell spike timestamps from all timestamps
    for ci in range(ncells):
        ts[ci] = ts_in[cids_in == cids_unique[ci]]
        cids[ci] = cids_unique[ci]

    return ts, cids


def spk_wform_prop2(cid, wf, polarity, do_skip_cluster_zero=True):
    """
    Calculate various spike waveform properties.
    This function is hardware-specific to DigitalLynx SX system.
    It expects 32 sample length and 32 kHz sampling frequency waveforms.
    """

    if not isinstance(cid, list):
        raise ValueError('spk_wform_prop2: cid must be a Nx1 list')

    if not isinstance(wf, list):
        raise ValueError('spk_wform_prop2: wf must be a Nx1 list')

    if len(cid) != len(wf):
        raise ValueError('spk_wform_prop2: cid and wf must be equal size')

    if not (isinstance(polarity, (int, float)) and not np.isnan(polarity) and not np.isinf(polarity) and polarity != 0):
        raise ValueError('spk_wform_prop2: polarity must be positive or negative scalar number')

    wf_ts_usec = np.linspace(0, 32 / 32000, 32) * 1e6
    wf_usts_usec = np.linspace(0, 32 / 32000, 128) * 1e6  # UpSampled ts
    ncells = len(wf)

    wfp = []

    for ci in range(ncells):
        cid_1cell = cid[ci]

        if cid_1cell == 0 and do_skip_cluster_zero:
            continue

        # wf_1cell = wf[ci]
        wf_1cell = np.array(wf[ci])

        if wf_1cell.ndim != 3:
            raise ValueError('spk_wform_prop2: wf_1cell must be a 3D matrix')

        if wf_1cell.shape[0] != 32:
            raise ValueError('spk_wform_prop2: wf_1cell must be 32xNxM matrix')

        wfp_1cell = spk_wform_prop_one_cell(wf_1cell, wf_ts_usec, wf_usts_usec, polarity)

        wfp.append({
            'cid': cid_1cell,
            'wf': wfp_1cell['wf'],
            'wf_std': wfp_1cell['wf_std'],
            'wf_sem': wfp_1cell['wf_sem'],
            'wf_all': wfp_1cell['wf_all'],
            'wf_good_idx': wfp_1cell['wf_good_idx'],
            'wf_us': wfp_1cell['wf_us'],
            'wf_us_std': wfp_1cell['wf_us_std'],
            'wf_us_sem': wfp_1cell['wf_us_sem'],
            'best_ch': wfp_1cell['best_ch'],
            'wf_bad_prop': wfp_1cell['wf_bad_prop'],
            'wf_swing': wfp_1cell['wf_swing'],
            'wf_peak': wfp_1cell['wf_peak'],
            'wf_width': wfp_1cell['wf_width'],
            'wf_amp_ass': wfp_1cell['wf_amp_ass'],
            'rms': wfp_1cell['rms']
        })

    return wfp


def spk_wform_prop_one_cell(x, wf_ts, wf_usts, polarity):
    out = {}

    nchan = x.shape[1]  # nchan == 1 for nse 2 for nst or 4 for ntt files.

    if nchan != 1:
        wf_per_ch = np.zeros((len(wf_ts), nchan))
        wf_swing_per_ch = np.zeros(nchan)

        for ch in range(nchan):
            wf_per_ch[:, ch] = np.mean(x[:, ch, :], axis=1)
            tmp_wf = wf_per_ch[:, ch]
            tmp = np.max(tmp_wf[tmp_wf > 0]) + np.max(np.abs(tmp_wf[tmp_wf < 0]))
            wf_swing_per_ch[ch] = 0 if tmp is None else tmp

        best_ch_idx = np.argmax(wf_swing_per_ch)
    else:
        best_ch_idx = 1

    wf_all = np.squeeze(x[:, best_ch_idx, :])

    wf_good, wf_bad_prop, wf_good_idx = spk_wform_filter(wf_all, polarity)
    wf_data_avg = np.mean(wf_good, axis=1)

    if np.any(wf_data_avg) == 0:
        raise ValueError('spk_wform_prop2: no waveform data')

    wf_data_std = np.std(wf_all, axis=1)
    wf_data_sem = wf_data_std / np.sqrt(wf_all.shape[1])

    # interp_func = interp1d(wf_ts, wf_data_avg, kind='spline')
    interp_func = UnivariateSpline(wf_ts, wf_data_avg, s=0)
    wf_us = interp_func(wf_usts)

    wf_amp_pos = np.max(wf_us[wf_us > 0])
    wf_amp_neg = np.max(np.abs(wf_us[wf_us < 0]))

    out['wf'] = wf_data_avg
    out['wf_std'] = wf_data_std
    out['wf_sem'] = wf_data_sem
    out['wf_all'] = wf_all
    out['wf_bad_prop'] = wf_bad_prop
    out['wf_good_idx'] = wf_good_idx

    # out['wf_us'] = wf_us
    # out['wf_us_std'] = interp1d(wf_ts, wf_data_std, kind='spline')(wf_usts)
    # out['wf_us_sem'] = interp1d(wf_ts, wf_data_sem, kind='spline')(wf_usts)

    # Interpolating wf_us_std using UnivariateSpline
    interp_func_std = UnivariateSpline(wf_ts, wf_data_std, s=0)  # s=0 means no smoothing
    wf_us_std = interp_func_std(wf_usts)

    # Interpolating wf_us_sem using UnivariateSpline
    interp_func_sem = UnivariateSpline(wf_ts, wf_data_sem, s=0)  # s=0 means no smoothing
    wf_us_sem = interp_func_sem(wf_usts)

    # Assign to the output dictionary
    out['wf_us'] = wf_us
    out['wf_us_std'] = wf_us_std
    out['wf_us_sem'] = wf_us_sem

    out['best_ch'] = best_ch_idx
    out['wf_swing'] = wf_amp_pos + wf_amp_neg

    if polarity > 0:
        out['wf_peak'] = wf_amp_pos
    else:
        out['wf_peak'] = wf_amp_neg

    out['wf_width'] = spk_wform_width(wf_us, wf_usts, polarity)

    if polarity > 0:
        out['wf_amp_ass'] = (wf_amp_neg - wf_amp_pos) / out['wf_swing']
    else:
        out['wf_amp_ass'] = (wf_amp_pos - wf_amp_neg) / out['wf_swing']

    out['rms'] = np.sqrt(np.sum(wf_us ** 2) / len(wf_us))

    return out


def spk_wform_width(wf_data, wf_ts, polarity):
    if polarity > 0:
        tsi_max = np.argmax(wf_data)
        tsi_valley = np.argmin(wf_data[tsi_max:]) + tsi_max

        if tsi_valley < 1 or tsi_valley > len(wf_data) or tsi_max < 1 or tsi_max > len(wf_data):
            print('spk_wform_prop2::spk_wform_width: ERROR, unable to estimate waveform width(1)')
            return np.nan

        return wf_ts[tsi_valley] - wf_ts[tsi_max]
    else:
        tsi_min = np.argmin(wf_data)
        tsi_valley = np.argmax(wf_data[tsi_min:]) + tsi_min

        if tsi_valley < 1 or tsi_valley > len(wf_data) or tsi_min < 1 or tsi_min > len(wf_data):
            print('spk_wform_prop2::spk_wform_width: ERROR, unable to estimate waveform width(2)')
            return np.nan

        return wf_ts[tsi_valley] - wf_ts[tsi_min]


def spk_wform_filter(wf_in, polarity):
    num_spk = wf_in.shape[1]

    if polarity > 0:
        tsi_pos = np.argmax(wf_in, axis=0)
        tsi_median = np.median(tsi_pos)
        idx_good = (tsi_pos <= (tsi_median + 4)) & (tsi_pos >= (tsi_median - 4))
    else:
        tsi_neg = np.argmin(wf_in, axis=0)
        tsi_median = np.median(tsi_neg)
        idx_good = (tsi_neg <= (tsi_median + 4)) & (tsi_neg >= (tsi_median - 4))

    wf_out = wf_in[:, idx_good]
    wf_bad = (num_spk - np.sum(idx_good)) / num_spk
    wf_bad = wf_bad * 100  # convert to percentage

    return wf_out, wf_bad, idx_good


def spk_select_wf(wf_in, cids_in, do_remove_cluster_zero=False):
    """
    SpkSelectWf provides follow up processing of spike waveforms extracted
    from sorted ntt/nst/nse files by using NlxGetSpikes* set of functions.
    """
    if wf_in.ndim != 3:
        raise ValueError('spk_select_wf: wf_in must be a 3D matrix')

    if wf_in.shape[0] != 32:
        raise ValueError('spk_select_wf: wf_in must be 32xNxM matrix')

    if not isinstance(cids_in, (list, np.ndarray)):
        raise ValueError('spk_select_wf: cids_in must be a vector')

    if wf_in.shape[2] != len(cids_in):
        raise ValueError('spk_select_wf: wf_in and cids_in must be equal length')

    cids_unique = np.unique(cids_in)  # unique cluster IDs

    if len(cids_unique) == 1 and cids_unique[0] == 0:
        raise ValueError('spk_select_wf: unsorted spikes provided')

    # remove cluster zero (unsorted spikes)
    if do_remove_cluster_zero:
        if cids_unique[0] == 0:
            cids_unique = cids_unique[1:]

    ncells = len(cids_unique)  # total number of cells (clusters) in the file

    # prepare storage for output
    wf = [None] * ncells
    cids = [None] * ncells

    # select per-cell spike waveforms from all waveforms
    for ci in range(ncells):
        wf[ci] = wf_in[:, :, cids_in == cids_unique[ci]]
        cids[ci] = cids_unique[ci]

    return wf


def spk_train_prop2(cid, spk_ts, spk_wf, spk_fet, wf_best_ch, trial_ts, polarity):
    """
    Calculate various spike train properties.
    This function is hardware-specific to DigitalLynx SX system.
    It expects 32 sample length and 32 kHz sampling frequency waveforms.
    """

    if not isinstance(cid, list):
        raise ValueError('spk_train_prop2: CID must be a list')

    if not isinstance(spk_ts, list):
        raise ValueError('spk_train_prop2: spk_ts must be a list')
    if not all(isinstance(x, list) for x in spk_ts):
        raise ValueError('spk_train_prop2: spk_ts must be a list of lists')

    if not isinstance(spk_wf, list):
        raise ValueError('spk_train_prop2: spk_wf must be a list')
    if not all(isinstance(x, list) for x in spk_wf):
        raise ValueError('spk_train_prop2: spk_wf must be a list of lists')

    if not isinstance(spk_fet, list):
        raise ValueError('spk_train_prop2: spk_fet must be a list')
    if not all(isinstance(x, list) for x in spk_fet):
        raise ValueError('spk_train_prop2: spk_fet must be a list of lists')

    if cid[0] != 0:
        raise ValueError('spk_train_prop2: first cluster must be cluster zero!')

    if len(cid) != len(wf_best_ch):
        raise ValueError('spk_train_prop2: lengths of wf_best_ch and cid must be equal')

    if len(trial_ts) != 2:
        raise ValueError('spk_train_prop2: trial_ts must be a 2 element array')

    if (trial_ts[1] - trial_ts[0]) <= 0:
        raise ValueError('spk_train_prop2: wrong trial timestamps?')

    if len(spk_ts) != len(spk_wf):
        raise ValueError('spk_train_prop2: spk_ts and spk_wf must be equal length')

    if len(spk_ts) != len(spk_fet):
        raise ValueError('spk_train_prop2: spk_ts and spk_fet must be equal length')

    if not isinstance(polarity, (int, float)) or np.isnan(polarity) or np.isinf(polarity) or polarity == 0:
        raise ValueError('spk_train_prop2: polarity must be positive or negative scalar number')

    wf_ts_usec = np.linspace(0, 32 / 32000, 32) * 1e6
    wf_usts_usec = np.linspace(0, 32 / 32000, 128) * 1e6

    num_spk_all = [len(ts) for ts in spk_ts]
    spk_blocks = np.cumsum(num_spk_all)
    spk_blocks = np.vstack((np.insert(spk_blocks[:-1], 0, 0), spk_blocks)).T

    spk_fet_all = np.hstack(spk_fet).T

    fet_sum = np.sum(spk_fet_all, axis=0)
    fet_zeros = np.where(fet_sum == 0)[0]
    if len(fet_zeros) == 2:
        spk_fet_all = np.delete(spk_fet_all, fet_zeros, axis=1)

    trp = []

    for cid_idx in range(len(spk_ts)):
        if cid[cid_idx] == 0:
            continue

        # cell_ts = spk_ts[cid_idx]
        cell_ts = np.array(spk_ts[cid_idx])
        # print(cell_ts.shape)
        # cell_wf = np.squeeze(spk_wf[cid_idx])[:, wf_best_ch[cid_idx]]
        cell_wf = np.array(np.squeeze(spk_wf[cid_idx])[:, wf_best_ch[cid_idx]])
        # print(len(cell_ts.shape))

        if (len(cell_ts.shape) != 1) or (cell_wf.shape[0] != 32) or (cell_ts.shape[0] != cell_wf.shape[1]):
            raise ValueError(f'Wrong size of cell_ts and/or cell_wf matrices for cell_id={cid_idx}')

        interp_func = interp1d(wf_ts_usec, cell_wf, kind='cubic', axis=0)
        cell_wf_us = interp_func(wf_usts_usec)

        cl = list(range(spk_blocks[cid_idx, 0], spk_blocks[cid_idx, 1]))

        num_fets = spk_fet_all.shape[1]
        num_spks = len(cl)
        if num_fets < num_spks:
            mahal_d = cluster_mahal(spk_fet_all, cl)
        else:
            mahal_d = np.nan

        print(trial_ts)
        trp.append({
            'num_spk': len(cell_ts),
            'frate_peak': 1 / np.min(medfilt(np.diff(cell_ts / 1e6), kernel_size=5)),
            'frate_mean': firing_rate(np.array(cell_ts) / 1e6, np.array([trial_ts]) / 1e6)[0],
            'perc_isi_u2ms': (len(np.where(np.diff(cell_ts) / 1e6 <= 0.002)[0]) * 100) / len(cell_ts),
            'csi_swing': calc_csi_swing(cell_ts, cell_wf_us),
            'csi_peaks': calc_csi_peaks(cell_ts, cell_wf_us, polarity),
            'lratio': lratio(mahal_d, spk_fet_all.shape[1], cl),
            'isold': isolation_distance(mahal_d, cl),
            'isi_mode_ms': mode(np.diff(cell_ts) / 1e3)
        })

    return trp


def calc_csi_swing(ts, wf):
    cell_wf_pos_idx = wf > 0
    cell_wf_neg_idx = wf < 0
    cell_wf_pos = np.where(cell_wf_neg_idx, 0, wf)
    cell_wf_neg = np.where(cell_wf_pos_idx, 0, wf)
    cell_wf_amp = np.max(cell_wf_pos, axis=0) + np.max(np.abs(cell_wf_neg), axis=0)

    if cell_wf_amp.shape[0] != 1:
        cell_wf_amp = cell_wf_amp.T

    return csi(ts / 1e6, cell_wf_amp)


def calc_csi_peaks(ts, wf, polarity):
    if polarity > 0:
        wf_peaks = np.max(wf, axis=0)
    else:
        wf_peaks = np.abs(np.min(wf, axis=0))

    if wf_peaks.shape[0] != 1:
        wf_peaks = wf_peaks.T

    return csi(ts / 1e6, wf_peaks)


def cluster_mahal(features, clusters=None):
    """
    Compute the Mahalanobis distance between feature vectors and clusters.
    """

    if features is None or not isinstance(features, np.ndarray) or features.ndim != 2 or features.size == 0:
        raise ValueError('Invalid features matrix')

    if clusters is None:
        clusters = [np.arange(features.shape[0])]
    elif isinstance(clusters, int):
        clusters = [np.array([clusters])]
    elif isinstance(clusters, list):
        if all(isinstance(c, int) for c in clusters):
            clusters = [np.array(clusters)]
        else:
            clusters = [np.array(c) if isinstance(c, (list, np.ndarray)) else np.array([c]) for c in clusters]
    elif isinstance(clusters, np.ndarray):
        if clusters.ndim == 1:
            clusters = [clusters]
        else:
            raise ValueError('Invalid index vector')
    else:
        raise ValueError('Invalid index vector')

    D2 = np.zeros((features.shape[0], len(clusters)))

    for k, cluster in enumerate(clusters):
        if cluster.size == 0:  # Equivalent to isempty in MATLAB
            D2[:, k] = np.nan
        else:
            subset = features[cluster, :]
            VI = np.linalg.inv(np.cov(subset.T)).T  # Inverse covariance matrix
            D2[:, k] = np.array([mahalanobis(f, subset.mean(axis=0), VI) for f in features])

    return D2

    #         cluster_features = features[cluster, :]
    #         mean_cluster = np.mean(cluster_features, axis=0)
    #         cov_cluster = np.cov(cluster_features, rowvar=False)
    #         inv_cov_cluster = inv(cov_cluster)

    #         for i, feature in enumerate(features):
    #             D2[i, k] = mahalanobis(feature, mean_cluster, inv_cov_cluster)

    # return D2


def firing_rate(events, segments=None):
    """
    Calculate mean firing rate.
    """
    if events is None:
        raise ValueError("Events must be provided")

    if isinstance(events, np.ndarray):
        events = [events]

    if not isinstance(events, list):
        raise ValueError("Invxalid events")

    n_events = len(events)

    if segments is None or len(segments) == 0:
        segments = np.array([[-np.inf, np.inf]])
        sl = np.array([max(event) - min(event) for event in events])
    elif segments.shape[1] != 2:
        raise ValueError("Invalid segments matrix")
    else:
        sl = np.diff(segments, axis=1).flatten()

    selection = seg_select(segments, events)
    n = np.array([len(sel) for sel in selection])
    fr = n / sl

    fr_total = np.sum(n) / np.sum(sl)

    return fr_total, fr


def seg_select(segments, events):
    """
    Select segments from events.
    """
    selection = []
    for event in events:
        sel = []
        for segment in segments:
            sel.append(event[(event >= segment[0]) & (event <= segment[1])])
        selection.append(np.concatenate(sel))
    return selection


def seg_select(segments, time, data=None, option='apart'):
    """
    Select all data within segments. ??
    """

    if segments.shape[1] != 2 or segments.shape[0] < 1 or not isinstance(segments, np.ndarray):
        raise ValueError("Invalid or empty list of segments")

    if isinstance(time, np.ndarray):
        time = [time]
    elif not isinstance(time, list):
        raise ValueError("Invalid time argument")

    if data is None:
        data = []
    elif isinstance(data, np.ndarray):
        data = [data]
    elif not isinstance(data, list):
        raise ValueError("Invalid data argument")

    if option not in ['apart', 'all']:
        raise ValueError("Invalid option argument")

    concatenate = option == 'all'

    selindex = [[None] * len(time) for _ in range(len(segments))]
    seltime = [[None] * len(time) for _ in range(len(segments))]
    seldata = [[None] * len(time) for _ in range(len(segments))]
    selnum = np.zeros((len(segments), len(time)), dtype=int)

    for k, t in enumerate(time):
        if len(t) == 0:
            continue

        for s in range(len(segments)):
            i = np.where((t >= segments[s, 0]) & (t <= segments[s, 1]))[0]
            selindex[s][k] = i

            if not concatenate:
                seltime[s][k] = t[i]
                if len(data) > 0:
                    seldata[s][k] = data[k][i]

            selnum[s][k] = len(i)

    if concatenate:
        idx = [None] * len(time)
        for k, t in enumerate(time):
            idx[k] = np.unique(
                np.concatenate([selindex[s][k] for s in range(len(segments)) if selindex[s][k] is not None]))
            seltime = [t[idx[k]] for k in range(len(time))]
            seldata = [data[k][idx[k]] for k in range(len(data))] if len(data) > 0 else []

        selindex = idx

    return seltime


def lratio(D2, df, clusters=None):
    """
    Compute L-ratio cluster quality measure.
    """
    if clusters is None:
        L = np.sum(1 - chi2.cdf(D2.ravel(), df))
        Lr = L / D2.size
    else:
        if isinstance(clusters, list) and all(isinstance(c, int) for c in clusters):
            clusters = [clusters]
        elif isinstance(clusters, np.ndarray) and clusters.ndim == 1:
            clusters = [clusters]

        L = np.zeros(len(clusters))
        Lr = np.zeros(len(clusters))
        for k, cluster in enumerate(clusters):
            idx = np.setdiff1d(np.arange(D2.shape[0]), cluster)
            L[k] = np.sum(1 - chi2.cdf(D2[idx, k], df))
            Lr[k] = L[k] / len(cluster)

    return Lr


def isolation_distance(D2, clusters):
    """
    Compute isolation distance measure.
    """
    # print(isinstance(clusters, list))
    # print(len(clusters))
    # print(D2.shape)

    if isinstance(clusters, list) and all(isinstance(c, int) for c in clusters):
        clusters = [clusters]
    elif isinstance(clusters, np.ndarray) and clusters.ndim == 1:
        clusters = [clusters]

    if isinstance(clusters, np.ndarray):
        clusters = [clusters]
        D2 = D2.ravel()
    elif not isinstance(clusters, list) or len(clusters) != D2.shape[1]:
        raise ValueError('Invalid index argument')

    nc = len(clusters)
    nr = D2.shape[0]
    D = np.full(nc, np.nan)

    # Loop through all clusters
    for k in range(nc):
        n = len(clusters[k])  # number of spikes in cluster

        # Continue if cluster has no spikes or if there are less spikes outside the cluster than inside
        if n == 0 or (nr - n) < n:
            continue

        # Find indices NOT belonging to cluster
        idx = np.setdiff1d(np.arange(D2.shape[0]), clusters[k])
        # Sort distances
        sortD2 = np.sort(D2[idx, k])

        # Find isolation distance
        D[k] = sortD2[n]

    return D


def csi(spike_times, spike_amp=None, interval=[0.003, 0.015]):
    """
    Calculate complex spike index.
    """

    if isinstance(spike_times, np.ndarray):
        spike_times = [spike_times]
    elif not isinstance(spike_times, list):
        raise ValueError('Invalid spike_times')

    spike_times = [np.array(st, dtype=float) for st in spike_times]

    n_spike_vectors = len(spike_times)

    if spike_amp is None:
        spike_amp = [np.ones(len(st)) for st in spike_times]
    elif isinstance(spike_amp, np.ndarray) and n_spike_vectors == 1:
        if len(spike_times[0]) != len(spike_amp):
            raise ValueError('Amplitude and spike time vectors have different lengths')
        spike_amp = [spike_amp]
    elif isinstance(spike_amp, list) and len(spike_amp) == n_spike_vectors:
        for i in range(n_spike_vectors):
            if len(spike_amp[i]) != len(spike_times[i]):
                raise ValueError('Amplitude and spike time vectors have different lengths')
    else:
        raise ValueError('Invalid spike amplitude vector')

    spike_amp = [np.array(sa, dtype=float) for sa in spike_amp]

    if not isinstance(interval, list) or len(interval) != 2:
        raise ValueError('Invalid max and min intervals')

    min_int, max_int = interval

    c = np.zeros((n_spike_vectors, n_spike_vectors))

    for i in range(n_spike_vectors):
        for j in range(n_spike_vectors):
            if i != j:
                dt, idx = isi(spike_times[i], spike_times[j], 'smallest')
                ii = ~np.isnan(idx)
                dt = -dt[ii]
                da = spike_amp[i][ii] - spike_amp[j][idx[ii]]
                c[i, j] = calccsi(dt, da, max_int, min_int) / len(spike_times[i])
            else:
                dt, idx = isi(spike_times[i], 'smallest')
                ii = ~np.isnan(idx)
                dt = -dt[ii]
                da = spike_amp[i][ii] - spike_amp[i][idx[ii]]
                c[i, j] = calccsi(dt, da, max_int, min_int) / len(spike_times[j])

    return c


def isi(event1, *args):
    """
    Return intervals for every event.
    """

    # Default interval type
    returntype = 'post'

    # Check event vector
    if not isinstance(event1, (list, np.ndarray)) or len(np.shape(event1)) != 1:
        raise ValueError('Invalid event times vector')

    event1 = np.array(event1).flatten()
    n = len(event1)

    if n == 0:
        return np.array([]), np.array([])

    # Check interval type
    if len(args) == 2 and isinstance(args[1], str):
        returntype = args[1]
        if returntype not in ['pre', '<', 'post', '>', 'smallest', 'largest']:
            raise ValueError('Invalid interval type')

    # Check if second event vector is present
    event2 = None
    if len(args) >= 1 and not isinstance(args[0], str):
        event2 = np.array(args[0]).flatten()
        if not isinstance(event2, (list, np.ndarray)) or len(np.shape(event2)) != 1:
            raise ValueError('Invalid event times vector')

        if len(event2) == 0:
            return np.full(n, np.nan), np.full(n, np.nan)
    else:
        event2 = event1

    m = len(event2)

    # Compute intervals
    if m == 0:
        i_pre = np.concatenate(([np.nan], -np.diff(event1)))
        ind_pre = np.concatenate(([np.nan], np.arange(n - 1)))
        i_post = np.concatenate((np.diff(event1), [np.nan]))
        ind_post = np.concatenate((np.arange(1, n), [np.nan]))
    else:
        ind_pre = np.floor(np.interp(event1, event2, np.arange(1, m + 1))).astype(int) - 1
        valids = ~np.isnan(ind_pre)
        i_pre = np.full(n, np.nan)
        if np.any(valids):
            i_pre[valids] = event2[ind_pre[valids]] - event1[valids]
            i_pre[np.argmax(valids):] = event2[-1] - event1[np.argmax(valids):]
            ind_pre[np.argmax(valids):] = m - 1

        ind_post = np.ceil(np.interp(event1, event2, np.arange(1, m + 1))).astype(int) - 1
        valids = ~np.isnan(ind_post)
        i_post = np.full(n, np.nan)
        if np.any(valids):
            i_post[valids] = event2[ind_post[valids]] - event1[valids]
            i_post[:np.argmin(valids)] = event2[0] - event1[:np.argmin(valids)]
            ind_post[:np.argmin(valids)] = 0

    # Assign outputs
    if returntype in ['pre', '<']:
        ii = i_pre
        idx = ind_pre
    elif returntype in ['post', '>']:
        ii = i_post
        idx = ind_post
    elif returntype == 'smallest':
        ii = np.where(np.abs(i_post) <= np.abs(i_pre), i_post, i_pre)
        ii[np.isnan(i_pre) | np.isnan(i_post)] = np.nan
        idx = np.where(np.abs(i_post) <= np.abs(i_pre), ind_post, ind_pre)
    elif returntype == 'largest':
        ii = np.where(np.abs(i_post) > np.abs(i_pre), i_post, i_pre)
        ii[np.isnan(i_pre) | np.isnan(i_post)] = np.nan
        idx = np.where(np.abs(i_post) > np.abs(i_pre), ind_post, ind_pre)
    else:
        raise ValueError('Invalid interval type')

    return ii, idx


def calccsi(dt, da, max_int, min_int):
    """
    Calculate complex spike index.
    """
    # Find all valid intervals (i.e., interval smaller than or equal to max_int)
    valid = np.abs(dt) <= max_int

    # Find intervals within refractory period
    refract = np.abs(dt) < min_int

    # Find intervals for all quadrants
    q1 = (da <= 0) & (dt > 0)  # post intervals with smaller amplitude
    q2 = (da > 0) & (dt < 0)  # pre intervals with larger amplitude
    q3 = (da <= 0) & (dt < 0)  # pre intervals with smaller amplitude
    q4 = (da > 0) & (dt > 0)  # post intervals with larger amplitude

    # Count the number of intervals that contribute positively to CSI
    pos = np.sum((q1 | q2) & valid & ~refract)

    # Count the number of intervals that contribute negatively to CSI
    neg = np.sum((q3 | q4 | refract) & valid)

    # Calculate CSI
    csi = 100 * (pos - neg)

    return csi


def mode(arr):
    values, counts = np.unique(arr, return_counts=True)
    m = counts.argmax()
    return values[m]


def calc_place_fields(param, behav, cts):
    """
    Calculate firing rate maps, place fields and other place cell related properties.
    """
    plf = []

    # Part 1: Process behavior data
    pos_x = np.array(behav['pos_x_cm'])
    pos_y = np.array(behav['pos_y_cm'])
    pos_ts = np.array(behav['pos_ts_usec'])
    vel = np.array(behav['vel_cmsec'])

    x_min = param['bhv_cx'] - param['bhv_hsz_x']
    x_max = param['bhv_cx'] + param['bhv_hsz_x'] + 1
    y_min = param['bhv_cy'] - param['bhv_hsz_y']
    y_max = param['bhv_cy'] + param['bhv_hsz_y'] + 1

    # Delete bad position values
    bad_idx = (pos_x > x_max) | (pos_x < x_min) | (pos_y > y_max) | (pos_y < y_min)
    pos_x = pos_x[~bad_idx]
    pos_y = pos_y[~bad_idx]
    pos_ts = pos_ts[~bad_idx]
    vel = vel[~bad_idx]

    # Extract intervals above velocity threshold
    hvel_interv = find_segments(vel, param['vel_thresh_cmsec'], int(behav['camera_fps']))
    hvel_ts = np.array([pos_ts[hvel_interv[:, 0]], pos_ts[hvel_interv[:, 1]]]).T
    pos_x_flt, pos_y_flt,_ = get_behav_per_trial(pos_x, pos_y, pos_ts, hvel_interv, pos_ts[0], pos_ts[-1])

    # Prepare colormap
    color_map = plt.cm.jet(param['plf_num_colors'] - int(param['plf_num_colors'] * 0.2))

    # Part 2: Process each cell
    for cid in range(len(cts)):
        # Interpolate position at which each spike occurred
        spk_pos_x, spk_pos_y, _, _ = get_spike_pos_per_trial(pos_x, pos_y, pos_ts, cts[cid], hvel_interv, pos_ts[0], pos_ts[-1])

        # Calculate occupancy map and firing rate map (raw and smoothed)
        rmap_smooth, ocmap_smooth = ratemap(spk_pos_x, spk_pos_y, pos_x_flt, pos_y_flt, [x_max - x_min, y_max - y_min], behav['camera_fps'])
        rmap_raw, ocmap_raw = ratemap(spk_pos_x, spk_pos_y, pos_x_flt, pos_y_flt, [x_max - x_min, y_max - y_min], behav['camera_fps'], smooth=False)

        rmap_smooth = rmap_smooth.T
        ocmap_smooth = ocmap_smooth.T
        rmap_raw = rmap_raw.T
        ocmap_raw = ocmap_raw.T

        # Calculate various place field properties
        frate_mean, frate_per_seg = firing_rate(cts[cid] / 1e6, hvel_ts / 1e6)
        frate_std = np.nanstd(frate_per_seg)
        spatinf_bspk = spatialinfo(rmap_smooth, ocmap_smooth)
        spatinf_bsec = spatialinfo(rmap_smooth, ocmap_smooth, bits_per_second=True)
        frate_peak = np.max(rmap_smooth)

        # Convert rate map to color map for plotting
        rmap_img = np.floor(param['plf_num_colors'] * rmap_smooth / frate_peak) + 2
        rmap_img, color_map = frmap_post_proc(rmap_img, color_map)

        # Detect place fields
        if param['plf_thr_type'] == 'perPFR':
            frate_thr = (param['plf_thr_perPFR'] / 100) * frate_peak
        elif param['plf_thr_type'] == 'numSD':
            frate_thr = frate_mean + param['plf_thr_numSD'] * frate_std
        else:
            raise ValueError('Unknown type of threshold')

        frf, frf_labels, frf_x, frf_y, frf_rate, frf_size = calc_place_field_info(rmap_smooth, frate_thr)

        # Calculate properties of the maximum firing rate place field
        if frf_rate.size > 0:
            max_frf_rate = np.max(frf_rate)
            max_frf_idx = np.argmax(frf_rate)
            max_frf_x = frf_x[max_frf_idx]
            max_frf_y = frf_y[max_frf_idx]
            max_frf_size = frf_size[max_frf_idx]

            rmap = rmap_raw[frf == max_frf_rate]
            smap = rmap_smooth[frf == max_frf_rate]
            if rmap.size > 0 and smap.size > 0:
                r = np.corrcoef(rmap, smap)[0, 1]
                coherence = 0.5 * (np.log(1 + r) - np.log(1 - r))
                if np.isinf(coherence):
                    coherence = np.nan
            else:
                coherence = 0

            in_plf = np.mean(rmap)
            tmp_map = np.copy(rmap_raw)
            tmp_map[frf == max_frf_rate] = np.nan
            tmp_map = tmp_map[~np.isnan(tmp_map)]
            out_plf = np.mean(tmp_map)
            snr = (in_plf - out_plf) / (in_plf + out_plf)

            max_frf_label = frf_labels[max_frf_y, max_frf_x]
            in_field_frate = np.nanmean(rmap_smooth[frf_labels == max_frf_label])

            out_field_pix = rmap_smooth[(frf_labels != max_frf_label) & (frf_labels != 0)]
            if out_field_pix.size > 0:
                out_field_frate = np.nanmean(out_field_pix)
            else:
                out_field_frate = 0

        else:
            max_frf_rate = 0
            max_frf_size = np.nan
            max_frf_x = np.nan
            max_frf_y = np.nan
            coherence = 0
            snr = 0
            in_field_frate = 0
            out_field_frate = 0

        rmap_smooth_nans = np.isnan(rmap_smooth)
        sampled_area = rmap_smooth.size - np.sum(rmap_smooth_nans)
        if sampled_area == 0:
            sampled_area = 1
        perc_whole_sz = (100 * max_frf_size) / sampled_area
        rmap_smooth_above_thr = rmap_smooth > frate_thr
        perc_pix_above_thr = (100 * np.sum(rmap_smooth_above_thr)) / sampled_area

        ocmap_smooth_no_nan = np.copy(ocmap_smooth)
        ocmap_smooth_no_nan[np.isnan(ocmap_smooth_no_nan)] = 0
        rmap_smooth_no_nan = np.copy(rmap_smooth)
        rmap_smooth_no_nan[rmap_smooth_nans] = 0

        occ_prob = ocmap_smooth_no_nan / np.sum(ocmap_smooth_no_nan)
        sparsity_2d = np.sum(occ_prob * rmap_smooth_no_nan) ** 2 / np.sum(occ_prob * rmap_smooth_no_nan ** 2)

        # Save calculated parameters
        plf.append({
            'rmap_smooth': rmap_smooth,
            'ocmap_smooth': ocmap_smooth,
            'rmap_raw': rmap_raw,
            'ocmap_raw': ocmap_raw,
            'rmap_img': rmap_img,
            'color_map': color_map,
            'spk_pos_x': spk_pos_x,
            'spk_pos_y': spk_pos_y,
            'frate_peak': frate_peak,
            'frate_mean': frate_mean,
            'frate_std': frate_std,
            'frate_thr': frate_thr,
            'spatinf_bspk': spatinf_bspk,
            'spatinf_bsec': spatinf_bsec,
            'max_frf_rate': max_frf_rate,
            'max_frf_x': max_frf_x,
            'max_frf_y': max_frf_y,
            'max_frf_size': max_frf_size,
            'coherence': coherence,
            'snr': snr,
            'in_field_frate': in_field_frate,
            'out_field_frate': out_field_frate,
            'perc_whole_sz': perc_whole_sz,
            'perc_pix_above_thr': perc_pix_above_thr,
            'sparsity_2d': sparsity_2d
        })

    return plf

def deltas(grid, dim=None):
    """
    Returns grid spacing.
    """
    if dim is None:
        dim = range(len(grid))
    elif isinstance(dim, int):
        dim = [dim]
    elif not all(isinstance(d, int) and 1 <= d <= len(grid) for d in dim):
        raise ValueError('Invalid dimension')

    dt = np.full(len(dim), np.nan)

    for i, d in enumerate(dim):
        if grid[d]['type'] == 'linear':
            if iscategorical(grid[d]):
                dt[i] = 1
            elif isuniform(grid[d]):
                dt[i] = np.mean(np.diff(centers(grid[d])))

    return dt



def find_segments(velocity, threshold, fps):
    """
    Find segments where velocity is above a threshold for at least 1 second.
    """
    high_vel = velocity > threshold
    segments = []
    start = None
    for i, val in enumerate(high_vel):
        if val and start is None:
            start = i
        elif not val and start is not None:
            if i - start >= fps:
                segments.append((start, i - 1))
            start = None
    if start is not None and len(high_vel) - start >= fps:
        segments.append((start, len(high_vel) - 1))
    return np.array(segments)


def get_behav_per_trial(pos_x, pos_y, ts, interv_idxs, trial_start, trial_stop):
    """
    Filters behavior data based on the intervals where the speed exceeds a threshold.
    """
    linear_good_idx = []

    for j in range(interv_idxs.shape[0]):
        linear_good_idx.extend(range(interv_idxs[j, 0], interv_idxs[j, 1] + 1))

    ts_flt = ts[linear_good_idx]
    pos_x_flt = pos_x[linear_good_idx]
    pos_y_flt = pos_y[linear_good_idx]

    good_idx = np.where((ts_flt >= trial_start) & (ts_flt <= trial_stop))[0]

    ts_flt = ts_flt[good_idx]
    pos_x_flt = pos_x_flt[good_idx]
    pos_y_flt = pos_y_flt[good_idx]


def iscategorical(grid, dim=None):
    """
    Checks whether grid is categorical or numerical.
    """
    if dim is None:
        dim = range(len(grid))
    elif isinstance(dim, int):
        dim = [dim]
    elif not all(isinstance(d, int) and 1 <= d <= len(grid) for d in dim):
        raise ValueError('Invalid dimension')

    b = np.zeros(len(dim))

    for i, d in enumerate(dim):
        if np.any(np.isnan(grid[d]['bins'])) or np.any(np.isinf(grid[d]['bins'])):
            b[i] = 1

    return b


def isuniform(grid, dim=None):
    """
    Test whether grid is uniform.
    """
    if dim is None:
        dim = range(len(grid))
    elif isinstance(dim, int):
        dim = [dim]
    elif not all(isinstance(d, int) and 1 <= d <= len(grid) for d in dim):
        raise ValueError('Invalid dimension')

    b = np.ones(len(dim))

    for i, d in enumerate(dim):
        if grid[d]['type'] == 'linear':
            ctrs = np.diff(np.diff(centers(grid, d)))
            bsz = binsizes(grid, d)
            bsz = bsz - np.mean(bsz)
            if (len(ctrs) > 0 and (np.any(np.isnan(ctrs)) or np.any(ctrs > 1e-10))) or \
                    (len(bsz) > 0 and (np.any(np.isnan(bsz)) or np.any(bsz > 1e-10))):
                b[i] = 0
        elif grid[d]['type'] == 'circular':
            raise NotImplementedError('Not implemented for circular grids')

    return b


def centers(grid, dim=None):
    """
    Get bin centers.

    Parameters:
    grid : list of dicts
        List where each element corresponds to a grid dimension with keys 'type' and 'bins'.
    dim : int or list of ints, optional
        Dimension(s) for which to get the bin centers. If None, get for all dimensions.

    Returns:
    list
        List of vectors with bin centers for each dimension.
    """
    if dim is None:
        dim = range(len(grid))
    elif isinstance(dim, int):
        dim = [dim]
    elif not all(isinstance(d, int) and 1 <= d <= len(grid) for d in dim):
        raise ValueError('Invalid dimension')

    val = []

    for d in dim:
        if grid[d]['type'] == 'linear':
            val.append(np.mean(grid[d]['bins'], axis=1))
        elif grid[d]['type'] == 'circular':
            val.append(limit2pi(circ_mean(np.column_stack([grid[d]['bins'][:-1], grid[d]['bins'][1:]]), axis=1)))

    return val if len(dim) > 1 else val[0]


def binsizes(grid, dim=None):
    """
    Get bin sizes.
    """
    if dim is None:
        dim = range(len(grid))
    elif isinstance(dim, int):
        dim = [dim]
    elif not all(isinstance(d, int) and 1 <= d <= len(grid) for d in dim):
        raise ValueError('Invalid dimension')

    val = []

    for d in dim:
        if grid[d]['type'] == 'linear':
            val.append(np.diff(grid[d]['bins'], axis=0))
        # elif grid[d]['type'] == 'circular':
        #     val.append(circ_diff(grid[d]['bins']))

    return val if len(dim) > 1 else val[0]


def circ_mean(alpha, w=None, dim=0):
    """
    Compute the mean direction for circular data.
    """
    if w is None:
        w = np.ones_like(alpha)
    elif w.shape != alpha.shape:
        raise ValueError('Input dimensions do not match')

    # Compute weighted sum of cos and sin of angles
    r = np.sum(w * np.exp(1j * alpha), axis=dim)

    # Obtain mean direction
    mu = np.angle(r)

    # # Confidence limits if desired
    # if nargout > 1:
    #     t = circ_confmean(alpha, 0.05, w, dim=dim)
    #     ul = mu + t
    #     ll = mu - t
    #     return mu, ul, ll

    return mu


def circ_r(alpha, w=None, d=0, dim=0):
    """
    Computes the mean resultant length for circular data.
    """
    alpha = np.asarray(alpha)
    if w is None:
        w = np.ones_like(alpha)
    else:
        w = np.asarray(w)
        if w.shape != alpha.shape:
            raise ValueError('Input dimensions do not match')

    # compute weighted sum of cos and sin of angles
    r = np.sum(w * np.exp(1j * alpha), axis=dim)

    # obtain length
    r = np.abs(r) / np.sum(w, axis=dim)

    # apply correction factor to correct for bias in estimation of r
    if d != 0:
        c = d / (2 * np.sin(d / 2))
        r = c * r

    return r


def circ_confmean(alpha, xi=0.05, w=None, d=0, dim=0):
    """
    Computes the confidence limits on the mean for circular data.
    """
    alpha = np.asarray(alpha)
    if w is None:
        w = np.ones_like(alpha)
    else:
        w = np.asarray(w)
        if w.shape != alpha.shape:
            raise ValueError('Input dimensions do not match')

    r = circ_r(alpha, w, d, dim)
    n = np.sum(w, axis=dim)
    R = n * r
    c2 = chi2.ppf(1 - xi, 1)

    t = np.zeros_like(r)

    for i in range(r.size):
        if r.flat[i] < 0.9 and r.flat[i] > np.sqrt(c2 / (2 * n.flat[i])):
            t.flat[i] = np.sqrt((2 * n.flat[i] * (2 * R.flat[i] ** 2 - n.flat[i] * c2)) / (4 * n.flat[i] - c2))
        elif r.flat[i] >= 0.9:
            t.flat[i] = np.sqrt(n.flat[i] ** 2 - (n.flat[i] ** 2 - R.flat[i] ** 2) * np.exp(c2 / n.flat[i]))
        else:
            t.flat[i] = np.nan
            print('Warning: Requirements for confidence levels not met.')

    t = np.arccos(t / R)

    return t


def limit2pi(angle):
    return np.mod(angle, 2 * np.pi)


def get_spike_pos_per_trial(pos_x_cm, pos_y_cm, pos_ts_usec, spts_usec, interv_idxs, trial_start_usec, trial_end_usec):
    """
    Interpolates spike positions and filters them based on specified intervals.
    """
    pos_x_cm = np.asarray(pos_x_cm)
    pos_y_cm = np.asarray(pos_y_cm)
    pos_ts_usec = np.asarray(pos_ts_usec)
    spts_usec = np.asarray(spts_usec)

    if pos_x_cm.ndim != 1 or pos_y_cm.ndim != 1 or pos_ts_usec.ndim != 1 or spts_usec.ndim != 1:
        raise ValueError("All input arrays must be 1-dimensional")

    spk_pos_x_interp = interp1d(pos_ts_usec, pos_x_cm, kind='linear', bounds_error=False, fill_value='extrapolate')(spts_usec)
    spk_pos_y_interp = interp1d(pos_ts_usec, pos_y_cm, kind='linear', bounds_error=False, fill_value='extrapolate')(spts_usec)

    spk_pos_x = []
    spk_pos_y = []
    spk_ts_usec = []
    good_idx_cum = []

    for j in range(interv_idxs.shape[0]):
        good_interv_start_ts = pos_ts_usec[interv_idxs[j, 0]]
        good_interv_end_ts = pos_ts_usec[interv_idxs[j, 1]]
        good_idxs = np.where((spts_usec >= good_interv_start_ts) & (spts_usec <= good_interv_end_ts))[0]

        if good_idxs.size > 0:
            good_idx_cum.extend(good_idxs)
            spk_pos_x.extend(spk_pos_x_interp[good_idxs])
            spk_pos_y.extend(spk_pos_y_interp[good_idxs])
            spk_ts_usec.extend(spts_usec[good_idxs])

    spk_pos_x = np.array(spk_pos_x)
    spk_pos_y = np.array(spk_pos_y)
    spk_ts_usec = np.array(spk_ts_usec)

    good_idxs = np.where((spk_ts_usec >= trial_start_usec) & (spk_ts_usec <= trial_end_usec))[0]

    if good_idxs.size > 0:
        spk_pos_x = spk_pos_x[good_idxs]
        spk_pos_y = spk_pos_y[good_idxs]
        spk_ts_usec = spk_ts_usec[good_idxs]

    return spk_pos_x, spk_pos_y, spk_ts_usec, np.array(good_idx_cum)

def smoothn(data, *args):
    """
    Smooth data with a Gaussian kernel.
    """
    options = {'correct': 0, 'nanexcl': 0, 'kernel': 'gauss', 'normalize': 1}
    options, other = parse_args(args, options)

    sd = []
    dx = []

    if other:
        sd = other[0]
        if len(other) > 1:
            dx = other[1]

    nd = data.ndim

    if options['kernel'] in ['none']:
        return data, None
    elif options['kernel'] in [None, 'gauss', 'normal']:
        if not sd:
            sd = np.ones(nd)
        elif np.isscalar(sd):
            sd = np.full(nd, sd)
        else:
            sd = np.asarray(sd)[:nd]

        if not dx:
            dx = np.ones(nd)
        elif np.isscalar(dx):
            dx = np.full(nd, dx)
        else:
            dx = np.asarray(dx)[:nd]

        if data.ndim == 1:
            if data.shape[0] == 1:  # row vector
                sd[0] = 0
                dx[0] = 1
            else:  # column vector
                sd[1] = 0
                dx[1] = 1

        kernel = gaussn(sd, dx)

    elif options['kernel'] in ['box']:
        if not sd:
            sd = np.full(nd, 3)
        elif np.isscalar(sd):
            sd = np.full(nd, sd)

        if not dx:
            dx = np.ones(nd)
        elif np.isscalar(dx):
            dx = np.full(nd, dx)

        if data.ndim == 1:
            if data.shape[0] == 1:  # row vector
                sd[0] = 0
                dx[0] = 1
            else:  # column vector
                sd[1] = 0
                dx[1] = 1

        npoints = np.round(sd / dx).astype(int)
        npoints[npoints == 0] = 1
        kernel = np.ones(npoints)

    else:
        if not isinstance(options['kernel'], np.ndarray):
            raise ValueError('Invalid kernel')
        kernel = options['kernel']

    if options['normalize']:
        kernel /= kernel.sum()

    if options['nanexcl']:
        idx = np.isnan(data)
        data[idx] = 0
        kernel[np.isnan(kernel)] = 0
    else:
        idx = np.array([])

    data = convolve(data, kernel, mode='constant')

    if options['correct']:
        n = convolve(np.ones(data.shape), kernel, mode='constant')
        n[idx] = 0
        data /= n

    return data, kernel

def gaussn(sd=1, dx=1, n=4):
    """
    Create a Gaussian kernel.
    """
    # Expand scalar inputs
    if np.isscalar(sd):
        sd = np.full_like(dx, sd) if not np.isscalar(dx) else np.full(1, sd)
    elif np.isscalar(dx):
        dx = np.full_like(sd, dx)
    elif len(sd) != len(dx):
        raise ValueError('Incompatible sizes of sd and dx vectors')

    # Remove zero standard deviations
    valid_sd = sd != 0
    nvalid = np.sum(valid_sd)
    perm_dim = np.argsort(~valid_sd)
    sd = sd[valid_sd]
    dx = dx[valid_sd]

    if np.any(np.isnan(sd)) or np.any(np.isinf(sd)) or np.any(dx == 0) or np.any(np.isnan(dx)) or np.any(np.isinf(dx)):
        raise ValueError('Invalid standard deviations and/or sample frequencies')

    # Make sure sd and dx are column vectors
    sd = np.atleast_1d(sd)
    dx = np.atleast_1d(dx)

    nd = len(sd)

    # Calculate size of kernel
    npoints = np.round(n * sd / dx).astype(int)

    # Construct N-D grid
    ranges = [np.linspace(-n * sd_i, n * sd_i, 2 * npoints_i + 1) for sd_i, npoints_i in zip(sd, npoints)]
    grids = np.meshgrid(*ranges, indexing='ij')
    shape = grids[0].shape
    grids = np.stack(grids, axis=-1).reshape(-1, nd)

    # Construct n-dimensional Gaussian kernel
    kernel = multivariate_normal.pdf(grids, mean=np.zeros(nd), cov=np.diag(sd ** 2))
    kernel = kernel.reshape(shape)

    if len(perm_dim) > 1:
        kernel = np.transpose(kernel, perm_dim)

    return kernel

def ratemap(spike_behavior, behavior, *args):
    """
    Create rate maps.
    """
    if spike_behavior is None or behavior is None:
        raise ValueError('Invalid arguments')

    options = {
        'samplefreq': 30,
        'grid': [],
        'smoothtype': 'pre',
        'smooth': [],
        'validitymask': [],
        'nandefault': True,
        'minoccupancy': 0,
        'normalize': True
    }

    options, other, remainder = parse_args(args, options)

    nosample_val = np.nan if options['nandefault'] else 0

    # Create occupancy map and grid
    om, g = map_behavior(behavior, *remainder, grid=options['grid'], default=nosample_val)

    # Get sample spacing in grid
    dx = deltas(g)
    dx[np.isnan(dx) | np.isinf(dx)] = 1

    # Set standard deviation to be the same for all dimensions if only one value is given
    if np.isscalar(options['smooth']):
        options['smooth'] = np.full(len(dx), options['smooth'])

    # Convert occupancy to seconds
    om = om / options['samplefreq']

    # Set invalid bins to NaN
    if len(options['validitymask']) > 0:
        if not np.array_equal(om.shape, options['validitymask'].shape):
            raise ValueError('Invalid sized mask')
        om[~options['validitymask']] = np.nan

    # Treat bins with low occupancy as not sampled
    low_occ = om <= options['minoccupancy']

    # Smooth if requested
    if options['smoothtype'] == 'pre' and len(options['smooth']) > 0 and np.any(options['smooth'] != 0):
        om[low_occ] = nosample_val
        om = smoothn(om, options['smooth'], dx, nanexcl=True, correct=True)

    # Compute spike behavior maps
    rm, _ = map_behavior(spike_behavior, *remainder, grid=g, default=0)

    if len(options['validitymask']) > 0:
        rm[~np.broadcast_to(options['validitymask'], rm.shape)] = np.nan

    # Smooth if requested
    if options['smoothtype'] == 'pre' and len(options['smooth']) > 0 and np.any(options['smooth'] != 0):
        rm[low_occ] = nosample_val
        rm = smoothn(rm, options['smooth'], dx, nanexcl=True, correct=True)

    # Compute rate map
    if options['normalize']:
        rm = np.divide(rm, om, where=om != 0)
        if options['smoothtype'] == 'post' and len(options['smooth']) > 0 and np.any(options['smooth'] != 0):
            rm[low_occ] = nosample_val
            rm = smoothn(rm, options['smooth'], dx, nanexcl=True, correct=True)

    return rm, g, om


def spatialinfo(spike, occ, method='rate', meanrate=None):
    """
    Calculate spatial information.
    """
    units = 'bits/spike'

    if method not in ['rate', 'prob']:
        raise ValueError('spatialinfo: invalid method')

    if method == 'rate':
        if meanrate is not None:
            units = 'bits/second'
    elif method == 'prob':
        if meanrate is not None:
            units = 'bits/second'
        else:
            raise ValueError('spatialinfo: meanrate must be provided for method "prob"')

    spike = np.asarray(spike)
    occ = np.asarray(occ)

    if spike.ndim < occ.ndim or spike.ndim > occ.ndim + 1:
        raise ValueError('spatialinfo: invalid spike and/or occupancy arrays')

    if spike.shape[:occ.ndim] != occ.shape:
        raise ValueError('spatialinfo: spike and occupancy arrays must have matching shapes')

    if occ.ndim == spike.ndim:
        spike = np.expand_dims(spike, axis=-1)

    if method == 'rate':
        nspikes = spike * occ
        totalspikes = np.nansum(nspikes, axis=tuple(range(occ.ndim)), keepdims=True)
        totaltime = np.nansum(occ)
        pspike = nspikes / totalspikes
        pocc = spike / (totalspikes / totaltime)
        si = np.nansum(pspike * np.log2(pspike / pocc), axis=tuple(range(occ.ndim)), keepdims=True)

        if units == 'bits/second':
            si *= totalspikes / totaltime
        si = np.squeeze(si)

    elif method == 'prob':
        if meanrate is None:
            raise ValueError('spatialinfo: meanrate must be provided for method "prob"')

        meanrate = np.asarray(meanrate)
        if meanrate.ndim != 1 or meanrate.size != spike.shape[-1]:
            raise ValueError('spatialinfo: invalid meanrate array')

        si = np.nansum(spike * np.log2(spike / occ), axis=tuple(range(occ.ndim)))

        if units == 'bits/second':
            si *= meanrate
        si = np.squeeze(si)

    return si


def frmap_post_proc(rmap_img, color_map):
    """
    Process firing rate map for better visualization.
    """
    rmap_img = np.asarray(rmap_img)
    color_map = np.asarray(color_map)

    # Replace NaNs with zeros
    rmap_img[np.isnan(rmap_img)] = 0

    # Smooth with 3x3 uniform kernel
    kernel = np.ones((3, 3)) / 8
    rmap_img_out = convolve(rmap_img, kernel, mode='constant', cval=0.0)

    # Shift up for using white background
    rmap_img_out += 1

    # Turn background to white
    color_map[0] = [1, 1, 1]

    return rmap_img_out, color_map


def calc_place_field_info(fratemap, frate_threshold):
    """
    Calculate place field information from the firing rate map.
    """
    if fratemap.ndim != 2 or not np.issubdtype(fratemap.dtype, np.number):
        raise ValueError('calc_place_field_info: Invalid matrix')

    nrows, ncols = fratemap.shape

    frfields = np.zeros((nrows, ncols))
    frlabels = np.zeros((nrows, ncols))

    field_x = []
    field_y = []
    field_rate = []
    field_size = []

    pk_x, pk_y, pk_z = localmaximum2d(fratemap, threshold=1)

    if len(pk_x) == 0:
        return frfields, frlabels, field_x, field_y, field_rate, field_size

    fields = fratemap > frate_threshold
    L, num = label(fields, structure=np.ones((3, 3)))
    frlabels = L

    for n in range(1, num + 1):
        tmp_idx = np.where(L == n)
        tmp_x, tmp_y = tmp_idx[1], tmp_idx[0]
        tmp_z = []

        for pkn in range(len(pk_x)):
            if pk_x[pkn] in tmp_x and pk_y[pkn] in tmp_y:
                tmp_z.append(pk_z[pkn])

        if tmp_z:
            max_z = max(tmp_z)
            idx_z = np.where(pk_z == max_z)[0]

            for j in idx_z:
                field_x.append(pk_x[j])
                field_y.append(pk_y[j])
                field_rate.append(pk_z[j])
                field_size.append(len(tmp_x))
                frfields[tmp_idx] = pk_z[j]

    return frfields, frlabels, field_x, field_y, field_rate, field_size

def localmaximum2d(m, threshold=-np.inf, mindist=0):
    if m.ndim != 2 or not np.issubdtype(m.dtype, np.number):
        raise ValueError('localmaximum2d: Invalid matrix')

    mcopy = np.copy(m)
    m[np.isnan(m)] = -np.inf

    coordinates = peak_local_max(m, min_distance=mindist, threshold_abs=threshold)

    x = coordinates[:, 1]
    y = coordinates[:, 0]
    z = mcopy[y, x]

    return x, y, z

def is_bad_value(value):
    """
    Checks if the input value is considered bad (empty, NaN, or non-numeric).
    """
    if value is None or np.isnan(value) or not isinstance(value, (int, float, np.number)):
        return True
    return False


def calc_cell_type_weak(wfp, stp):
    """
    Calculate the type of cell (unit) based on its waveform properties (WFP)
    and spike train properties (STP).
    """
    if len(wfp) != len(stp):
        raise ValueError('calc_cell_type_weak: size of all input structures must be the same')

    ct = []

    for cid in range(len(wfp)):
        if (is_bad_value(stp[cid]['num_spk'])) or \
                (is_bad_value(stp[cid]['frate_mean'])) or \
                (is_bad_value(stp[cid]['perc_isi_u2ms'])) or \
                (is_bad_value(stp[cid]['csi_swing'])) or \
                (is_bad_value(stp[cid]['csi_peaks'])) or \
                (is_bad_value(stp[cid]['lratio'])) or \
                (is_bad_value(wfp[cid]['wf_width'])):
            ct.append('BAD')
            continue

        if stp[cid]['num_spk'] < 50:
            ct.append('LOW_NOS')
            continue

        if stp[cid]['frate_mean'] >= 10:
            ct.append('INN_1')
            continue

        if stp[cid]['perc_isi_u2ms'] >= 0.5:
            ct.append('SHORT_ISI')
            continue

        if (stp[cid]['num_spk'] >= 50 and
                stp[cid]['perc_isi_u2ms'] < 0.5 and
                wfp[cid]['wf_width'] >= 200 and
                stp[cid]['csi_swing'] >= 5 and
                stp[cid]['isold'] >= 10):
            ct.append('PYR')
            continue

        if (stp[cid]['num_spk'] >= 50 and
                stp[cid]['perc_isi_u2ms'] < 0.5 and
                wfp[cid]['wf_width'] >= 170 and
                stp[cid]['csi_swing'] >= 5 and
                stp[cid]['isold'] >= 10):
            ct.append('PYR_nWf')
            continue

        ct.append('NA')

    return ct


def frmap_plot_stack(h_fig, AX, M, atl, do_adjust_cbar, font_size):
    """
    Plot a stack of firing rate maps with linked axes.

    Parameters:
    h_fig (matplotlib.figure.Figure): The figure to plot on.
    AX (list): List of axis handles.
    M (list of numpy arrays): List of firing rate maps.
    atl (str): Axis to link ('x' or 'y').
    do_adjust_cbar (bool): Whether to adjust the colorbar position.
    font_size (int): Font size for axis labels.
    """

    if not isinstance(AX, list) or not all(isinstance(ax, plt.Axes) for ax in AX):
        raise ValueError('AX must be a list of axis handles')

    if not isinstance(M, list) or not all(isinstance(m, np.ndarray) for m in M):
        raise ValueError('M must be a list of firing rate maps')

    if atl not in ['x', 'y']:
        raise ValueError('Unknown axis to link')

    if len(AX) != (len(M) + 1):
        raise ValueError('numel(AX) must be equal to numel(M) + 1')

    num_colors = 64
    kernel = np.ones((3, 3)) / 8
    plf_colormap = plt.get_cmap('jet', num_colors - int(num_colors * 0.2))
    plf_colormap.set_bad(color='white')  # turn background to white

    M_sz = np.array([m.shape for m in M])
    atl_idx = np.argsort(M_sz[:, 1 if atl == 'x' else 0])[::-1]
    global_max_frate = max(np.nanmax(m) for m in M)

    for ii in range(len(M)):
        M_tmp = M[ii]
        M_tmp = np.floor(num_colors * M_tmp / global_max_frate) + 2
        M_tmp[np.isnan(M_tmp)] = 0
        M_tmp = convolve(M_tmp, kernel, mode='constant')
        M_tmp += 1  # shift up for using white background
        M[ii] = M_tmp

    h_fig.clf()  # clear figure

    for ii in range(1, len(AX)):
        ax = AX[ii]
        ax.clear()
        im = ax.imshow(M[ii - 1], aspect='auto', cmap=plf_colormap, origin='lower')
        ax.set_yticks([])
        ax.set_xticks([])
        ax.tick_params(axis='both', which='major', labelsize=font_size)

    for idx in atl_idx:
        AX[idx + 1].get_shared_x_axes().join(AX[idx + 1], AX[1])
        AX[idx + 1].get_shared_y_axes().join(AX[idx + 1], AX[1])

    h_cbar = h_fig.colorbar(im, ax=AX[1:], orientation='vertical')
    h_cbar.set_ticks(np.linspace(0, num_colors, 5))
    h_cbar.set_ticklabels(np.round(np.linspace(0, global_max_frate, 5), 2))
    h_cbar.ax.tick_params(labelsize=font_size)

    if do_adjust_cbar:
        h_cbar.ax.set_position([0.8314, 0.1100, 0.0581, 0.8150])
        for ii in range(1, len(AX)):
            ax_pos = AX[ii].get_position()
            AX[ii].set_position([ax_pos.x0, ax_pos.y0, 0.85 * ax_pos.width, ax_pos.height])

    return h_fig, AX, h_cbar


def main():
    dset_list = load_targets('trials_tmaze.txt')
    s_fname_csv = 'place_field_linearized.csv'
    s_fname_suff_plf = '_place_field_linearized.mat'
    s_fname_bhv = 'bhv_linearized.json'

    dsp_delay = 984
    bin_sz_cm = 2

    lcnt = 1
    list_out = []

    for did in dset_list:
        did = did.strip()
        print(f'Process dataset: {did}')

        out_dir = os.path.join(did, f'plf_whole_trial-{datetime.now().strftime("%Y-%m-%d-%H-%M")}/')
        os.makedirs(out_dir, exist_ok=True)

        did_path = Path(did)
        s_trial = did_path.name
        s_dataset = did_path.parent.name
        s_mouse = did_path.parent.parent.name
        s_group = did_path.parent.parent.parent.name

        # Create the figure for the 2D firing map plot
        h_fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Load behavioral nvt data for the dataset.
        trg1 = os.path.join(did, s_fname_bhv)
        with open(trg1, 'r') as file:
            BHV = json.load(file)

        ts_trial = [BHV['BEHAV']['pos_ts_usec'][0], BHV['BEHAV']['pos_ts_usec'][-1]]

        # read and handle ntt files
        spike_data = []
        current_cell_id = 0
        excitatory_neurons = []
        inhibitory_neurons = []
        tetrode_cell_ids = []

        ntt_file_list = load_target_flist(did, 'cut_TT*.json')
        for ntt_file in ntt_file_list:
            print(f'\tProcess file: {ntt_file}')

            # Extracting the tetrode number from filename
            match = re.search(r'cut_TT(\d+)\.json', ntt_file)
            if match:
                ntt_file_num = match.group(1)
            else:
                print("No match found")

            # Load corresponding NCS files
            ncs_file_name = load_target_flist(did, f'*CSC{ntt_file_num}_FSI*.json')

            # Initialize lists for data processing
            lfp_wband = []
            lfp_ts_usec = []
            lfp_srate = []
            lfp_theta = []

            # Load NCS data
            with open(ncs_file_name[0], 'r') as file:
                ncs_json = json.load(file)
            print('NCS_json.keys:', ncs_json.keys())

            lfp_wband = np.array(ncs_json['data_samples']).reshape(1, -1)
            lfp_ts_usec = np.array(ncs_json['ts_usec']).reshape(1, -1)
            # lfp_srate = ncs_json['lFP_srate']
            lfp_srate = ncs_json['freq']

            # Example parameters
            fs = lfp_srate  # Sampling frequency, e.g., 1000 Hz
            theta_band = PARAM['theta_band']  # Theta band frequencies, e.g., (6, 10)
            numtaps = 101  # Number of filter taps (coefficients), should be odd

            # Filter the LFP data within the theta band
            # lfp_theta = np.apply_along_axis(least_squares_bandpass_filter, 1, lfp_wband, fs, theta_band, numtaps)

            lfp_theta = eegfilt(lfp_wband, fs, PARAM['theta_band'][0], PARAM['theta_band'][1])

            # Load NTT data
            with open(ntt_file, 'r') as file:
                ntt_json = json.load(file)
            print('NTT_json.keys:', ntt_json.keys())

            all_ts = np.array(ntt_json['time'][0]) - dsp_delay
            # all_cids = pd.DataFrame(ntt_json['cell_id']).values  # Convert to numpy array
            all_cids = np.array(ntt_json['cell_id'][0])
            # all_wfs = pd.DataFrame(ntt_json['waveform']).values  # Convert to numpy array
            all_wfs = np.array(ntt_json['waveform']).T
            # all_fets = pd.DataFrame(ntt_json['feature']).T.to_numpy()  # Convert to numpy array
            all_fets = np.array(ntt_json['feature']).T

            # all_ts, all_cids, all_wfs, all_fets, _ = NlxGetSpikesAll('cut_TT2.ntt')

            cell_fet, _ = spk_select_fet(all_fets, all_cids)

            cell_ts, cell_ids = spk_select_ts(all_ts, all_cids)
            # print(len(cell_ids))

            cell_wfs = spk_select_wf(all_wfs, all_cids)
            # print(len(cell_wfs))

            # Identify and remove bad cells
            bad_cells = [len(ts) < 50 for ts in cell_ts]
            cell_fet = [fet for fet, bad in zip(cell_fet, bad_cells) if not bad]
            cell_fet = [fet.tolist() for fet in cell_fet]
            cell_ids = [cid for cid, bad in zip(cell_ids, bad_cells) if not bad]
            # print(len(cell_ids))
            cell_ts = [ts for ts, bad in zip(cell_ts, bad_cells) if not bad]
            # print("I cannot see you ")
            cell_ts = [ts.tolist() for ts in cell_ts]
            # print(cell_ts)
            cell_wfs = [wf for wf, bad in zip(cell_wfs, bad_cells) if not bad]
            # print(len(cell_wfs))
            cell_wfs = [wf.tolist() for wf in cell_wfs]
            # print(cell_ts)
            # print(cell_wfs)

            # calculations for all cells in each ntt file, output
            # WFP，STP，place field and CT for all cells in the ntt file
            # calculation 1：Calculate waveform properties
            wfp = spk_wform_prop2(cell_ts, cell_wfs, 1, True)
            # wfp_best_ch = [wfp[0]['best_ch']] + [w['best_ch'] for w in wfp]
            wfp_best_ch = [w['best_ch'] for w in wfp]
            # print(len([wfp[0]['best_ch']]))
            # print(len([w['best_ch'] for w in wfp]))
            # print(len(wfp_best_ch))

            # calculation 2：Calculate spike train properties
            stp = spk_train_prop2(cell_ids, cell_ts, cell_wfs, cell_fet, wfp_best_ch, ts_trial, 1)
            # remove cluster zero before place field calculation
            cell_fet = cell_fet[1:]
            cell_ids = cell_ids[1:]
            cell_ts = cell_ts[1:]
            cell_wfs = cell_wfs[1:]
            # calculation 3：Calculate place fields etc.
            plf = calc_place_fields(BHV['PARAM'], BHV['BEHAV'], cell_ts)
            # calculation 4：Cell types
            ct = calc_cell_type_weak(wfp, stp)

            data = {
                'cell_ts': [ts.tolist() for ts in cell_ts],  # Convert numpy arrays to lists if necessary
                'cell_ids': cell_ids,  # Assuming cell_ids is a list
                'WFP': wfp,  # Assuming wfp is a list of dicts or similar structure
                'STP': stp,  # Assuming stp is a list of dicts or similar structure
                'PLF': plf,  # Assuming plf is a list of dicts or similar structure
                's_fname_csv': s_fname_csv,  # Replacing 's_*' with its actual name
                'CT': ct,  # Assuming ct is a list of strings or similar structure
                'bin_sz_cm': bin_sz_cm,  # Assuming bin_sz_cm is a numerical value
                'LFP_TS_USEC': lfp_ts_usec.tolist() if isinstance(lfp_ts_usec, np.ndarray) else lfp_ts_usec,
                'LFP_WBAND': lfp_wband.tolist() if isinstance(lfp_wband, np.ndarray) else lfp_wband,
                'LFP_theta': lfp_theta.tolist() if isinstance(lfp_theta, np.ndarray) else lfp_theta,
                'LFP_srate': lfp_srate  # Assuming lfp_srate is a numerical value
            }

            # Construct the JSON filename
            json_filename = ntt_file_list[ntt_file][:-4] + s_fname_suff_plf.replace('.mat', '.json')

            # Save the data as JSON
            with open(json_filename, 'w') as json_file:
                json.dump(data, json_file, indent=4)

            # loop by number of cell to save data for each cell in the ntt
            list_out = []
            for cid in range(len(plf)):  # loop & output to list_out
                print(f'\tProcess cell: {cell_ids[cid]}')

                s_out = f"{s_dataset},{s_trial},{ntt_file_list[ntt_file]},{cell_ids[cid]},{s_mouse},{s_group},"
                s_out += f"{ct[cid]},"

                # WFP...
                s_out += f"{wfp[cid]['wf_bad_prop']:.3f},"
                s_out += f"{wfp[cid]['wf_peak']:.3f},"
                s_out += f"{wfp[cid]['wf_swing']:.3f},"
                s_out += f"{wfp[cid]['wf_width']:.3f},"
                s_out += f"{wfp[cid]['wf_amp_ass']:.3f},"
                s_out += f"{wfp[cid]['rms']:.3f},"

                # STP...
                s_out += f"{stp[cid]['num_spk']},"
                s_out += f"{stp[cid]['frate_peak']:.4f},"
                s_out += f"{stp[cid]['frate_mean']:.4f},"
                s_out += f"{stp[cid]['perc_isi_u2ms']:.4f},"
                s_out += f"{stp[cid]['csi_swing']:.4f},"
                s_out += f"{stp[cid]['csi_peaks']:.4f},"
                s_out += f"{stp[cid]['lratio']:.6f},"
                s_out += f"{stp[cid]['isold']:.4f},"

                # PLF...
                s_out += f"{len(plf[cid]['spk_pos_x'])},"
                s_out += f"{plf[cid]['frate_peak']:.4f},"
                s_out += f"{plf[cid]['frate_mean']:.4f},"
                s_out += f"{plf[cid]['frate_std']:.4f},"
                s_out += f"{plf[cid]['frate_thr']:.4f},"
                s_out += f"{plf[cid]['spatinf_bspk']:.4f},"
                s_out += f"{plf[cid]['spatinf_bsec']:.4f},"
                s_out += f"{plf[cid]['max_frf_x']:.4f},"
                s_out += f"{plf[cid]['max_frf_y']:.4f},"
                s_out += f"{plf[cid]['max_frf_size']:.4f},"
                s_out += f"{plf[cid]['coherence']:.4f},"
                s_out += f"{plf[cid]['snr']:.4f},"
                s_out += f"{plf[cid]['in_field_frate']:.4f},"
                s_out += f"{plf[cid]['out_field_frate']:.4f},"
                s_out += f"{plf[cid]['perc_whole_sz']:.4f},"
                s_out += f"{plf[cid]['perc_pix_above_thr']:.4f},"
                s_out += f"{plf[cid]['sparsity_2d']:.4f},"

                s_out += '\n'
                list_out.append(s_out)
                lcnt += 1

                s_fname_out = f"{os.path.splitext(ntt_file_list[ntt_file])[0]}_cell{cell_ids[cid]}_{ct[cid]}"

                # Save 2D firing map plot as JPEG
                h_fig, ax, _ = frmap_plot_stack(h_fig, ax, [plf[cid]['rmap_smooth']], 'x', False, 8)
                h_fig.set_size_inches(20, 20)
                h_fig.savefig(f"{out_dir}/{s_fname_out}.jpg", dpi=300, format='jpeg')
                plt.show()

    # Save the spike data as JSON
    json_data = {
        "Spike_Data": spike_data.tolist() if isinstance(spike_data, np.ndarray) else spike_data,
        "Excitatory_Neurons": excitatory_neurons,
        "Inhibitory_Neurons": inhibitory_neurons,
        "Tetrode_Cell_IDs": tetrode_cell_ids
    }

    json_filename = 'Spike_Data_Processed.json'
    with open(json_filename, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

    # Save list_out to CSV
    csv_filename = s_fname_csv
    with open(csv_filename, 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerow([
            'Dataset', 'Trial', 'File', 'Cell_ID', 'Mouse', 'Group', 'Cell_TYPE',
            '% Bad WF', 'Peak mV', 'Swing mV', 'Width usec', 'Amp. Assym', 'RMS',
            'train_Nspk', 'train_FRpeak', 'train_FRmean', 'train % ISI < 2ms',
            'train_CSI_swing', 'train_CSI_peaks', 'train_Lratio', 'train_Isol.D',
            'PLF_Nspk', 'PLF_FRpeak', 'PLF_FRmean', 'PLF_SD.', 'PLF_Thr.',
            'Spat. info (bit/spikes)', 'Spat. info (bit/sec)', 'Max.FRF_X',
            'Max.FRF_Y', 'Max.FRF_Size', 'Coherence', 'SNR', 'In-field FR',
            'Out-field FR', '% sampled size', '% bins above thr.', '2D sparsity'
        ])
        for row in list_out:
            writer.writerow([row])


if __name__ == "__main__":
    main()

