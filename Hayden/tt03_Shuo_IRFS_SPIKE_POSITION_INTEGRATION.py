import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import json
import scipy.io as sio

def shuo_irfs_spike_position_integration(position_data, spike_data, minimum_time_difference=np.inf):
    position_data = np.array(position_data)
    spike_data = np.array(spike_data)

    def match_spike(n):
        time_diff = np.abs(position_data[:, 0] - spike_data[n, 0])
        min_diff = np.min(time_diff)
        index = np.where(time_diff == min_diff)[0][0]
        return position_data[index, :5]

    matched_spike_data = np.array(Parallel(n_jobs=-1)(delayed(match_spike)(n) for n in range(spike_data.shape[0])))

    time_difference = np.abs(spike_data[:, 0] - matched_spike_data[:, 0])
    spike_information = matched_spike_data[:, 1:5]  # Assuming these columns are X, Y, Direction, Velocity
    valid_indices = time_difference <= minimum_time_difference
    spike_information = spike_information[valid_indices, :]

    # Convert to DataFrame
    df = pd.DataFrame(spike_information, columns=['X_Position', 'Y_Position', 'Direction', 'Velocity'])

    # Format each float to 4 decimal places
    df = df.applymap(lambda x: float(f"{x:.4f}"))

    spike_data_to_save = {'Spike_Information': df.to_numpy().tolist()}
    with open('Spike_Information.json', 'w') as f:
        json.dump(spike_data_to_save, f, indent=4)

    return df


# with open('Spike_Data_Processed.json', 'r') as file:
#     Spike_Data_Processed = json.load(file)   
# Spike_Data_array = np.array(Spike_Data_Processed['Spike_Data'])


# with open('Position_Data_Processed.json', 'r') as file:
#     Position_Data_Processed = json.load(file)
# Position_Data = np.array(Position_Data_Processed['Position_Data'])

# # Call the function
# Spike_Information = shuo_irfs_spike_position_integration(Position_Data, Spike_Data_array)

