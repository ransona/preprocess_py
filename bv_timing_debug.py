import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d
from scipy.io import loadmat

import matplotlib.pyplot as plt

import organise_paths
import pickle

# os.environ['DISPLAY'] = 'localhost:10.0'

def run_preprocess_bv(userID, expID,plot_on):
    print('Starting run_preprocess_bv...')
    animalID, remote_repository_root, \
    processed_root, exp_dir_processed, \
        exp_dir_raw = organise_paths.find_paths(userID, expID)
    exp_dir_processed_recordings = os.path.join(processed_root, animalID, expID,'recordings')
    # load the stimulus parameter file produced by matlab by the bGUI
    # this includes stim parameters and stimulus order
    # try:
    #     stim_params = loadmat(os.path.join(exp_dir_raw, expID + '_stim.mat'))
    # except:
    #     raise Exception('Stimulus parameter file not found - this experiment was probably from pre-Dec 2021.')

    # load timeline
    Timeline = loadmat(os.path.join(exp_dir_raw, expID + '_Timeline.mat'))
    Timeline = Timeline['timelineSession']
    # get timeline file in a usable format after importing to python
    tl_chNames = Timeline['chNames'][0][0][0][0:]
    tl_daqData = Timeline['daqData'][0,0]
    tl_time    = np.squeeze(Timeline['time'][0][0])
    tl_bv_ch = np.where(np.isin(tl_chNames, 'Bonvision'))
    tl_bv_sync = np.squeeze(tl_daqData[:, tl_bv_ch])
    tl_pd_ch = np.where(np.isin(tl_chNames, 'Photodiode'))
    tl_pd = np.squeeze(tl_daqData[:, tl_pd_ch])

    frame_events = pd.read_csv(os.path.join(exp_dir_raw, expID + '_FrameEvents.csv'), names=['Frame', 'Timestamp', 'Sync', 'Trial'],
                            header=None, skiprows=[0], dtype={'Frame':np.float32, 'Timestamp':np.float32, 'Sync':np.float32, 'Trial':np.float32})

    # # Add running trace
    Encoder = pd.read_csv(os.path.join(exp_dir_raw, expID + '_Encoder.csv'), names=['Frame', 'Timestamp', 'Trial', 'Position'],
                            header=None, skiprows=[0]) #, dtype={'Frame':np.float32, 'Timestamp':np.float32, 'Trial':np.int64, 'Position':np.int64})
    bv_wheel_pos = np.squeeze(Encoder.Position.values)
    bv_wheel_pos_normed = (bv_wheel_pos-np.min(bv_wheel_pos))/(np.max(bv_wheel_pos)-np.min(bv_wheel_pos))
    bv_wheel_timestamps = Encoder.Timestamp.values

    # /////////////// DETECTING TIMING PULSES ///////////////
    # Find BV times when digital flips
    Timestamp = frame_events['Timestamp'].values
    Sync = frame_events['Sync'].values
    Trial = frame_events['Trial']
    Frame = frame_events['Frame']
    if Sync[0] == 1:
        print('Sync starts high so detecting falling edge')
        sync_polarity = -1
    else:
        print('Sync starts low so detecting rising edge')
        sync_polarity = 1

    flip_times_bv = np.squeeze(Timestamp[np.where((np.diff(Sync) == sync_polarity))[0]])

    # Find TL times when digital flips
    bv_ch = np.where(np.isin(tl_chNames, 'Bonvision'))
    tl_dig_thresholded = np.squeeze((tl_daqData[:, bv_ch] > 2.5).astype(int))
    flip_times_tl = np.squeeze(tl_time[np.where(np.diff(tl_dig_thresholded) == sync_polarity)])

    # find the moments when the (bonsai) trial number increments
    trialOnsetTimesBV = Timestamp[np.where(np.diff(Trial)==1)]
    # add in first trial onset
    trialOnsetTimesBV = np.insert(trialOnsetTimesBV,0,Timestamp[0])  

    # find times when PD goes high (trial start)
    tl_pd_thresholded = (tl_pd > 3).astype(int)
    tl_pd_trial_start_times = np.squeeze(tl_time[np.where(np.diff(tl_pd_thresholded) == 1)])

    # //////////////////////////////////////
    # Make model to convert BV time to TL time
    min_pulses = np.min([len(flip_times_tl),len(flip_times_bv)])
    linear_interpolator = interp1d(flip_times_bv[:min_pulses], flip_times_tl[:min_pulses], kind='linear', fill_value="extrapolate")
    trialOnsetTimesBV_tl = linear_interpolator(trialOnsetTimesBV)
    flip_times_bv_tl = linear_interpolator(flip_times_bv)
    # //////////////////////////////////////
    # Plotting  
    if plot_on:
        plt.figure()
        plt.subplot(2,1,1)
        # show waveform of a specific pulse detected in TL and BV
        trial_number = 0 # trial number to use for viewing x position of aligned signals
        pulse_num = 0 # pulse to use for initial alignment tl bv alignment
        use_pd = False
        
        pulse_time_tl = flip_times_tl[pulse_num]
        pulse_time_bv = flip_times_bv[pulse_num]
        tl_relative_time = tl_time - pulse_time_tl
        bv_relative_time = Timestamp - pulse_time_bv
        plt.plot(np.squeeze(tl_relative_time[::50]), np.squeeze(tl_dig_thresholded[::50]),label='BV in TL')
        plt.plot(np.squeeze(bv_relative_time), np.squeeze(Sync)+1.1,label='BV in BV')
        plt.plot(np.squeeze(tl_relative_time[::50]), (np.squeeze(tl_pd[::50])/3)+2.2,label='PD in TL')
        # plt.plot(np.squeeze(bv_wheel_timestamps - pulse_time_bv), bv_wheel_pos_normed+3.3,label='Wheel in BV')
        
        plt.scatter(tl_pd_trial_start_times - pulse_time_tl, np.ones(tl_pd_trial_start_times.shape)*-0.1,color='b',marker='*')
        plt.scatter(trialOnsetTimesBV_tl - pulse_time_tl, np.ones(trialOnsetTimesBV.shape)*-0.1,color='g',marker='o')
        plt.scatter(trialOnsetTimesBV - pulse_time_bv, np.ones(trialOnsetTimesBV.shape)*-0.1,color='r',marker='x')
        window_size = 30 # secs 
        if use_pd:
            plot_start = tl_pd_trial_start_times[trial_number] - (window_size/2) - pulse_time_tl
            plt.xlim(plot_start, plot_start+window_size)
        else:
            plot_start = trialOnsetTimesBV[trial_number] - (window_size/2) - pulse_time_bv
            plt.xlim(plot_start, plot_start+window_size)

        plt.title(f'BV and TL signals aligned using pulse {pulse_num} and showing trial {plot_start}\n{expID}')
        plt.legend() 

        # plot PD timed trial starts vs. BV trial number increment timed trial starts
        plt.subplot(2,1,2)
        if use_pd:
            # compare PD trial start times with BV trial number increment times
            drift_per_hour = np.max((tl_pd_trial_start_times-tl_pd_trial_start_times[0])-(trialOnsetTimesBV-trialOnsetTimesBV[0])) / (tl_pd_trial_start_times[-1] - tl_pd_trial_start_times[0]) * 3600
            plt.plot((tl_pd_trial_start_times-tl_pd_trial_start_times[0])-(trialOnsetTimesBV-trialOnsetTimesBV[0]),label=f'drift per hour = {drift_per_hour}')
            plt.legend()    
            plt.title('Relative times of PD hi and BV trial increment derived trial onsets')
        else:
            # compare TL captured sync pulse times with BV captured sync pulse times
            flip_count_diff = len(flip_times_tl) - len(flip_times_bv)
            min_flips = np.min([np.squeeze(len(flip_times_tl)),np.squeeze(len(flip_times_bv))])
            drift_per_hour              = ((flip_times_tl[min_flips-1]-flip_times_tl[0]) - (flip_times_bv[min_flips-1]-flip_times_bv[0])) / (flip_times_tl[-1] - flip_times_tl[0]) * 3600
            drift_per_hour_corrected    = ((flip_times_tl[min_flips-1]-flip_times_tl[0]) - (flip_times_bv_tl[min_flips-1]-flip_times_bv_tl[0])) / (flip_times_tl[-1] - flip_times_tl[0]) * 3600

            plt.plot((flip_times_tl[0:min_flips-1]-flip_times_tl[0]) - (flip_times_bv[0:min_flips-1]-flip_times_bv[0]),label=f'Raw: drift/hr={drift_per_hour}, flip diff={flip_count_diff}')
            plt.plot((flip_times_tl[0:min_flips-1]-flip_times_tl[0]) - (flip_times_bv_tl[0:min_flips-1]-flip_times_bv_tl[0]),label=f'Corrected: drift/hr={drift_per_hour_corrected}, flip diff={flip_count_diff}')
            plt.legend()  
            plt.title('Relative times of TL recorded BV flips and BV recorded BV flips')

        plt.show(block=False)
        x = 0

    min_len = np.min([np.squeeze(len(flip_times_tl)),np.squeeze(len(flip_times_bv))])
    initial_time_offset = np.median(flip_times_bv[0:10] - flip_times_tl[0:10])
    final_time_offset = np.median(flip_times_bv[min_len-10:min_len] - flip_times_tl[min_len-10:min_len])
    drift = (final_time_offset - initial_time_offset) / (flip_times_tl[min_len-1] - flip_times_tl[0]) * 3600

    return drift    

    # for debugging:
def main():
    userID = 'pmateosaparicio'
    # 2024-10-01_03_ESPM113 Drift in offset / hour = -996.2737655984166
    # 2024-10-02_01_ESPM113 Drift in offset / hour = -1067.8224571722203
    # 2024-10-04_01_ESPM113 
    # 2024-10-07_01_ESPM113 
    # 2024-10-08_01_ESPM113 
    # 2024-10-09_01_ESPM113 
    # 2024-10-12_02_ESPM113 
    # 2024-10-14_02_ESPM113 
    # 2024-10-17_01_ESPM113 
    # 2024-11-07_02_ESPM113 
    # 2024-11-08_02_ESPM115  
    # 2024-11-11_01_ESPM115 
    # 2024-11-12_01_ESPM115 
    # 2024-11-21_01_ESPM118 
    # 2024-11-27_01_ESPM118 
    # 2024-11-28_01_ESPM118 
    # 2024-12-05_01_ESPM117 

    allExp = ['2024-10-01_03_ESPM113','2024-10-02_01_ESPM113','2024-10-04_01_ESPM113','2024-10-07_01_ESPM113'
              ,'2024-10-08_01_ESPM113','2024-10-09_01_ESPM113','2024-10-12_02_ESPM113','2024-10-14_02_ESPM113'
              ,'2024-10-17_01_ESPM113','2024-11-07_02_ESPM113','2024-11-08_02_ESPM115','2024-11-11_01_ESPM115'
              ,'2024-11-12_01_ESPM115','2024-11-21_01_ESPM118','2024-11-27_01_ESPM118','2024-11-28_01_ESPM118'
              ,'2024-12-05_01_ESPM117']
    
    # 2024-12-10_01_ESPM118
    # 2024-12-10_02_ESPM118
    # allExp = ['2024-12-10_01_ESPM118','2024-12-10_02_ESPM118']

    allExp = ['2024-12-13_14_TEST']
    for expID in allExp:
        drift = run_preprocess_bv(userID, expID,plot_on=True)
        print(expID + ', ' + str(drift))


if __name__ == "__main__":
    main()