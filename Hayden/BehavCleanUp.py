import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

def BehavCleanUp(TS, Xpos, Ypos, params=None):
    """
    Cleanup behavior data from artifacts.
    Use this function BEFORE using behav_preproc.
    
    Parameters:
    TS : numpy.array
        Timestamps of the data points.
    Xpos : numpy.array
        X coordinates of the data points.
    Ypos : numpy.array
        Y coordinates of the data points.
    params : dict, optional
        Parameters for cleaning up the data.
        
    Returns:
    TSo, Xo, Yo : numpy.array
        Cleaned and interpolated timestamps, X coordinates, and Y coordinates.
    """

    # Convert arrays to float 
    TS = np.array(TS, dtype=float)
    Xpos = np.array(Xpos, dtype=float)
    Ypos = np.array(Ypos, dtype=float)

    # Ensure input arrays are column vectors
    TS = np.atleast_2d(TS).T
    Xpos = np.atleast_2d(Xpos).T
    Ypos = np.atleast_2d(Ypos).T

    # Default parameters
    if params is None:
        params = {
            'ds_thr': 10,
            'sgolayfilt_k': 3,
            'sgolayfilt_f': 31,
            'interp1_method': 'cubic'  # Changed to 'cubic' as 'v5cubic' is MATLAB specific
        }

    # Calculate differences and distances
    dX = np.diff(Xpos, axis=0)
    dY = np.diff(Ypos, axis=0)
    dS = np.sqrt(dX**2 + dY**2)
    dS = np.vstack([0, dS])  # Prepend 0 to make the array sizes match

    # Identify indices with large displacements
    ds_thr_idx = dS >= params['ds_thr']
    Xpos[ds_thr_idx] = np.nan
    Ypos[ds_thr_idx] = np.nan
    TS[ds_thr_idx] = np.nan

    # Smoothing using Savitzky-Golay filter
    Xo = savgol_filter(Xpos.flatten(), params['sgolayfilt_f'], params['sgolayfilt_k'], mode='nearest')
    Yo = savgol_filter(Ypos.flatten(), params['sgolayfilt_f'], params['sgolayfilt_k'], mode='nearest')

    # Interpolate NaNs
    def interpolate_nans(data, method='cubic'):
        non_nan_indices = ~np.isnan(data)
        interpolator = interp1d(np.flatnonzero(non_nan_indices), data[non_nan_indices], kind=method, fill_value="extrapolate")
        return interpolator(np.arange(len(data)))

    Xo = interpolate_nans(Xo, params['interp1_method'])
    Yo = interpolate_nans(Yo, params['interp1_method'])
    TSo = interpolate_nans(TS.flatten(), params['interp1_method'])

    return TSo, Xo, Yo

# # Generate some synthetic data
# np.random.seed(0)
# TS = np.linspace(0, 1, 100)
# Xpos = np.sin(2 * np.pi * TS) + np.random.normal(0, 0.1, TS.size)  # Sinusoidal signal with noise
# Ypos = np.cos(2 * np.pi * TS) + np.random.normal(0, 0.1, TS.size)

# # Introduce some artifacts
# Xpos[30:40] = np.nan  # Missing values
# Ypos[70:80] = np.nan  # Missing values

# TSo, Xo, Yo = BehavCleanUp(TS, Xpos, Ypos)

# # Plotting
# plt.figure(figsize=(12, 6))

# # Original Data
# plt.subplot(1, 2, 1)
# plt.plot(TS, Xpos, label='Xpos (Original)')
# plt.plot(TS, Ypos, label='Ypos (Original)')
# plt.title("Original Data")
# plt.xlabel("Time")
# plt.ylabel("Position")
# plt.legend()

# # Cleaned Data
# plt.subplot(1, 2, 2)
# plt.plot(TSo, Xo, label='Xo (Cleaned)')
# plt.plot(TSo, Yo, label='Yo (Cleaned)')
# plt.title("Cleaned Data")
# plt.xlabel("Time")
# plt.ylabel("Position")
# plt.legend()

# plt.tight_layout()
# plt.show()