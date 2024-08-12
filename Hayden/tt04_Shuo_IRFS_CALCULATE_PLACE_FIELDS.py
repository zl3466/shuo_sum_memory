import numpy as np
import json
from joblib import Parallel, delayed
import scipy.io as sio
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d
from scipy.signal import filtfilt
from scipy.signal import convolve
from scipy.signal.windows import gaussian


def main():
    with open('Spike_Data_Processed.json', 'r') as file:
        Spike_Data_Processed = json.load(file)   
    Spike_Data = np.array(Spike_Data_Processed['Spike_Data'])

    with open('Position_Data_Processed.json', 'r') as file:
        Position_Data_Processed = json.load(file)
    Position_Data = np.array(Position_Data_Processed['Position_Data'])

    with open('Spike_Information.json', 'r') as file:
        Spike_Information = json.load(file)   
    Spike_Information = np.array(Spike_Information['Spike_Information'])

    Bin_Size = 1
    Velocity_Cutoff = 10
    Firing_Rate_Cutoff = 0
    Analyze_Linear = 1

    Pos_x_value = Position_Data[:, 1].copy() 
    Spike_x_value = Spike_Information[:, 0].copy()

    Spike_Information[:, 0] = Spike_Information[:, 0] - np.min(Position_Data[:, 1]) + 0.001
    Spike_Information[:, 1] = Spike_Information[:, 1] - np.min(Position_Data[:, 2]) + 0.001

    Position_Data[:, 1] = Position_Data[:, 1] - np.min(Position_Data[:, 1]) + 0.001  
    Position_Data[:, 2] = Position_Data[:, 2] - np.min(Position_Data[:, 2]) + 0.001  

    Time_Change = np.diff(Position_Data[:, 0])
    Time_Change = np.append(Time_Change, Time_Change[-1]) # time gap between each behavior sample
    # eliminate zero or negative values by replacing them with a small positive number derived from the data itself
    Time_Change[Time_Change <= 0] = np.min(Time_Change[Time_Change > 0]) / 10

    X_Movement = np.diff(Position_Data[:, 1])
    X_Movement = np.append(X_Movement, X_Movement[-1])

    Y_Movement = np.diff(Position_Data[:, 2])
    Y_Movement = np.append(Y_Movement, Y_Movement[-1])

    Position_Change = np.column_stack((X_Movement, Y_Movement))

    Velocity = Position_Data[:, 4]  # calculate speed

    num_spikes = Spike_Data.shape[0]
    Spike_Movement = np.zeros((num_spikes, 2))

    # Loop over each spike entry
    for N in range(num_spikes):
        # Calculate the minimal absolute difference index
        differences = np.abs(Position_Data[:, 0] - Spike_Data[N, 0])
        min_diff = np.min(differences)
        index = np.where(differences == min_diff)[0][0]  # Take the first index directly
        # Assign the corresponding X and Y movements
        Spike_Movement[N, :] = [X_Movement[index], Y_Movement[index]]

    xposition_Cutoff = 34
    Index = (Velocity >= Velocity_Cutoff) & (Pos_x_value >= xposition_Cutoff)

    Position_Data = Position_Data[Index]
    Time_Change = Time_Change[Index]
    Position_Change = Position_Change[Index, :]
    Position_Data[:, 1] = np.ceil(Position_Data[:, 1] / Bin_Size)
    Position_Data[:, 2] = np.ceil(Position_Data[:, 2] / Bin_Size)

    Index = (Spike_Information[:, 3] >= Velocity_Cutoff) & (Spike_x_value >= xposition_Cutoff)  # do the same for spike data
    Spike_Data = Spike_Data[Index, :]
    Spike_Information = Spike_Information[Index, :]
    Spike_Information[:, 0] = np.ceil(Spike_Information[:, 0] / Bin_Size)
    Spike_Information[:, 1] = np.ceil(Spike_Information[:, 1] / Bin_Size)
    Spike_Movement = Spike_Movement[Index, :]

    # Initialize Time_In_Position matrix
    max_y = max(np.max(Position_Data[:, 2]), np.max(Spike_Information[:, 1]))
    max_x = max(np.max(Position_Data[:, 1]), np.max(Spike_Information[:, 0]))
    Time_In_Position = np.zeros((int(max_y), int(max_x)))

    # Populate Time_In_Position
    for N in range(Position_Data.shape[0]):
        y, x = Position_Data[N, 2], Position_Data[N, 1]
        Time_In_Position[int(y)-1, int(x)-1] += Time_Change[N]
        
    # Initialize Spikes_In_Position matrix
    max_z = np.max(Spike_Data[:, 1])
    Spikes_In_Position = np.zeros((int(max_y), int(max_x), int(max_z)))

    # Populate Spikes_In_Position
    for N in range(Spike_Data.shape[0]):
        y, x, z = Spike_Information[N, 1], Spike_Information[N, 0], Spike_Data[N, 1]
        Spikes_In_Position[int(y)-1, int(x)-1, int(z)-1] += 1


    if Analyze_Linear:
        # Check if the track is horizontal
        if (np.max(Position_Data[:, 1]) - np.min(Position_Data[:, 1])) >= (np.max(Position_Data[:, 2]) - np.min(Position_Data[:, 2])):
            Linear_Spikes_In_Position = np.sum(Spikes_In_Position, axis=0)  # Sum along y-direction
            Linear_Time_In_Position = np.sum(Time_In_Position, axis=0)  # Sum along y-direction
            Linear_Time_In_Position = np.transpose(Linear_Time_In_Position)
        
        # Check if the track is vertical
        elif (np.max(Position_Data[:, 1]) - np.min(Position_Data[:, 1])) < (np.max(Position_Data[:, 2]) - np.min(Position_Data[:, 2])):
            Linear_Spikes_In_Position = np.sum(Spikes_In_Position, axis=1)  # Sum along x-direction
            Linear_Time_In_Position = np.sum(Time_In_Position, axis=1)

        Linear_Rate_In_Position = np.zeros((Linear_Spikes_In_Position.shape[0], Linear_Spikes_In_Position.shape[1]))
        
        # Calculate firing rate
        for N in range(Linear_Spikes_In_Position.shape[1]):
            mask = Linear_Time_In_Position == 0
            Linear_Rate_In_Position[:, N] = 0
            Linear_Rate_In_Position[~mask, N] = Linear_Spikes_In_Position[~mask, N] / Linear_Time_In_Position[~mask]
            Linear_Rate_In_Position[:, N] = np.nan_to_num(Linear_Rate_In_Position[:, N], nan=0.0, posinf=0.0, neginf=0.0)        
        
        # Assuming 'Linear_Rate_In_Position' is already defined as a NumPy array
        Smoothed_Linear_Firing_Rate = Linear_Rate_In_Position.copy()
        
        sigma = 2  # Standard deviation for the Gaussian kernel
        
        Smoothed_Linear_Firing_Rate
        # Applying the filter to each column of the array
        for N in range(Linear_Rate_In_Position.shape[1]):
            Smoothed_Linear_Firing_Rate[:,N] = gaussian_filter1d(Linear_Rate_In_Position[:, N], sigma=sigma)

        # Replace NaN values with zero
        Smoothed_Linear_Firing_Rate[np.isnan(Smoothed_Linear_Firing_Rate)] = 0

        # Set negative values to zero
        Smoothed_Linear_Firing_Rate[Smoothed_Linear_Firing_Rate < 0] = 0
        
        for Z in range(Smoothed_Linear_Firing_Rate.shape[1]):
            if np.max(Smoothed_Linear_Firing_Rate[:, Z]) < Firing_Rate_Cutoff:
                Smoothed_Linear_Firing_Rate[:, Z] = 0
                
        # Assuming 'Smoothed_Linear_Firing_Rate' and 'Spike_Data' are defined and are NumPy arrays
        Field_Data_Linear = Smoothed_Linear_Firing_Rate.copy()

        # Initialize Firing_Rate_Peaks
        max_spike_data = int(np.max(Spike_Data[:, 1]))  # Adjusted for 0-based indexing
        Firing_Rate_Peaks = np.zeros((max_spike_data, 4))
        Firing_Rate_Peaks[:, 0] = np.arange(1, max_spike_data + 1)  # Column indices starting from 1

        # Populate the second column of Firing_Rate_Peaks with the index of max value in each column of Field_Data_Linear
        for N in range(max_spike_data):
            # Find the index of the maximum value in each column, using np.argmax which is zero-based
            # Ensure the index N is within the column range of Field_Data_Linear
            if N < Field_Data_Linear.shape[1]:
                Firing_Rate_Peaks[N, 1] = np.argmax(Field_Data_Linear[:, N]) + 1  # Convert to 1-based index for compatibility
                
                
                
        # Assuming 'Field_Data_Linear' and 'Firing_Rate_Peaks' are already defined
        data_dict = {
            'Field_Data_Linear': Field_Data_Linear.tolist(),  # Convert array to list
            'Firing_Rate_Peaks': Firing_Rate_Peaks.tolist()
        }

        # Save to .json file
        with open('Field_Data.json', 'w') as json_file:
            json.dump(data_dict, json_file)
        
if __name__ == "__main__":
    main()
        