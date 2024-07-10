import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

def NEW_BehavPreProc(pos_ts_usec, pos_x_cm, pos_y_cm, ang, nSDsm_pos, nSDsm_vel):
    """
    Preprocess behavioral data by smoothing positions and calculating velocities.
    
    Parameters:
        pos_ts_usec (array): Timestamps of position in microseconds.
        pos_x_cm (array): X coordinates in centimeters.
        pos_y_cm (array): Y coordinates in centimeters.
        ang (array): Head angles in degrees.
        nSDsm_pos (float): Standard deviation multiplier for position smoothing.
        nSDsm_vel (float): Standard deviation multiplier for velocity smoothing.

    Returns:
        dict: A dictionary containing smoothed position data, timestamps, head angles,
              camera frames per second, an outline of the position data, and calculated velocities.
    """

    behav = {}

    # Ensure inputs are column vectors to maintain consistency in data structure.
    pos_ts_usec = np.atleast_2d(pos_ts_usec).T
    pos_x_cm = np.atleast_2d(pos_x_cm).T
    pos_y_cm = np.atleast_2d(pos_y_cm).T

    # Determine the window length for Savitzky-Golay filter; ensure it's odd and at least 5.
    window_length = max(5, int(nSDsm_pos * np.std(pos_x_cm)) | 1)

    # Apply Savitzky-Golay filter to smooth x coordinates.
    behav['pos_x_cm'] = savgol_filter(pos_x_cm.flatten(), window_length, 3)  # polynomial order 3

    # Apply Savitzky-Golay filter to y coordinates if there's meaningful variation.
    if np.sum(np.diff(pos_y_cm)) > 0:
        window_length_y = max(5, int(nSDsm_pos * np.std(pos_y_cm)) | 1)
        behav['pos_y_cm'] = savgol_filter(pos_y_cm.flatten(), window_length_y, 3)
    else:
        behav['pos_y_cm'] = pos_y_cm.flatten()

    # Store timestamps and head angles.
    behav['pos_ts_usec'] = pos_ts_usec.flatten()
    behav['head_angles'] = ang

    # Calculate frames per second from the timestamps to help with frame-wise analysis.
    behav['camera_fps'] = 1e6 / np.mean(np.diff(behav['pos_ts_usec']))

    # Calculate positional boundaries, avoiding initial and final segments to reduce boundary effects.
    idx_1f = int(behav['camera_fps'])
    if len(behav['pos_x_cm']) > 2 * idx_1f:
        outline = {
            'Xmin': np.min(behav['pos_x_cm'][idx_1f:-idx_1f]) - 2,
            'Xmax': np.max(behav['pos_x_cm'][idx_1f:-idx_1f]) + 2,
            'Ymin': np.min(behav['pos_y_cm'][idx_1f:-idx_1f]) - 2,
            'Ymax': np.max(behav['pos_y_cm'][idx_1f:-idx_1f]) + 2
        }
    else:
        outline = {
            'Xmin': np.min(behav['pos_x_cm']) - 2,
            'Xmax': np.max(behav['pos_x_cm']) + 2,
            'Ymin': np.min(behav['pos_y_cm']) - 2,
            'Ymax': np.max(behav['pos_y_cm']) + 2
        }
    behav['outline'] = outline

    # Compute velocity from position changes over time.
    dX = np.diff(behav['pos_x_cm'])
    dY = np.diff(behav['pos_y_cm'])
    if len(dX) > 0 and len(dY) > 0:
        vel_cmsec = np.sqrt(dX**2 + dY**2) / (np.diff(behav['pos_ts_usec']) / 1e6)
        vel_cmsec = np.concatenate(([0], vel_cmsec))  # Assume initial velocity is zero
        vel_window_length = max(5, int(nSDsm_vel * np.std(vel_cmsec)) | 1)
        behav['vel_cmsec'] = savgol_filter(vel_cmsec, vel_window_length, 3)  # Smoothing the velocity
    else:
        behav['vel_cmsec'] = np.array([0])  # Default velocity if no data to differentiate

    return behav


# # Example usage
# pos_ts_usec = np.linspace(0, 1000000, 100)  # Example timestamps
# pos_x_cm = np.sin(np.linspace(0, 2 * np.pi, 100)) * 10  # Example X positions
# pos_y_cm = np.cos(np.linspace(0, 2 * np.pi, 100)) * 10  # Example Y positions
# ang = np.linspace(0, 360, 100)  # Example angles

# behav_data = BehavCleanUp(pos_ts_usec, pos_x_cm, pos_y_cm, ang, 0.5, 0.5)

# # Outputs
# behav_data['pos_x_cm'], behav_data['pos_y_cm'], behav_data['vel_cmsec']



# # Extract the data for plotting
# timestamps = pos_ts_usec.flatten()
# x_positions = behav_data['pos_x_cm']
# y_positions = behav_data['pos_y_cm']
# velocities = behav_data['vel_cmsec']

# # Plotting positions and velocities
# plt.figure(figsize=(14, 8))

# # Plot X and Y positions
# plt.subplot(2, 1, 1)
# plt.plot(timestamps, x_positions, label='Smoothed X Positions', color='blue')
# plt.plot(timestamps, y_positions, label='Smoothed Y Positions', color='red')
# plt.title('Smoothed Positions Over Time')
# plt.xlabel('Time (microseconds)')
# plt.ylabel('Position (cm)')
# plt.legend()

# # Plot velocities
# plt.subplot(2, 1, 2)
# plt.plot(timestamps, velocities, label='Velocity (cm/sec)', color='green')  # Corrected timestamps for velocity
# plt.title('Velocity Over Time')
# plt.xlabel('Time (microseconds)')
# plt.ylabel('Velocity (cm/sec)')
# plt.legend()

# plt.tight_layout()
# plt.show()