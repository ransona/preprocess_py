import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d
from scipy.io import loadmat
import matplotlib.pyplot as plt
import organise_paths
import pickle
import harp

# os.environ['DISPLAY'] = 'localhost:10.0'

def bv_timing_bug2(userID, expID,plot_on):
    animalID, remote_repository_root, \
    processed_root, exp_dir_processed, \
        exp_dir_raw = organise_paths.find_paths(userID, expID)
    exp_dir_processed_recordings = os.path.join(processed_root, animalID, expID,'recordings')

    # load Harp raw data
    harp_data_path = os.path.join(exp_dir_raw, expID + '_Behavior_Event44.bin')
    data_read = harp.io.read(harp_data_path)
    data_read_np = np.array(data_read)
    data_read_np[0:400,0] = data_read_np[400,0]
    harp_pd = data_read_np[:,0]
    harp_pd_time = np.arange(0, len(harp_pd)/1000, 1/1000)

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

    tl_pd2_ch = np.where(np.isin(tl_chNames, 'EyeCamera'))
    tl_pd2 = np.squeeze(tl_daqData[:, tl_pd2_ch])   


    # load bonsai data
    frame_events = pd.read_csv(os.path.join(exp_dir_raw, expID + '_FrameEvents.csv'), names=['Frame', 'Timestamp', 'Sync', 'Trial'],
                            header=None, skiprows=[0], dtype={'Frame':np.float32, 'Timestamp':np.float32, 'Sync':np.float32, 'Trial':np.float32})

    # plt.figure()
    # plt.plot(harp_pd_time+13.35,harp_pd/max(harp_pd),label='Harp PD',color='r')
    # plt.plot(tl_time,tl_pd/max(tl_pd),label='TL PD',color='b')
    # plt.plot(tl_time+0.07,tl_bv_sync/max(tl_bv_sync),label='TL BV',color='k')
    # plt.show(block=False) 

    # /////////////// DETECTING TIMING PULSES ///////////////

    # Find Harp times when PD flip
    #  Threshold the Harp PD signal and detect flips
    harp_pd_smoothed = pd.Series(harp_pd).rolling(window=20, min_periods=1).mean().values
    harp_pd_high = np.percentile(harp_pd_smoothed, 99)
    harp_pd_low = np.percentile(harp_pd_smoothed, 1)   
    harp_pd_threshold = harp_pd_low + ((harp_pd_high - harp_pd_low)*0.7)
     
    harp_pd_thresholded = np.where(harp_pd_smoothed < harp_pd_threshold, 0, 1)
    transitions = np.diff(harp_pd_thresholded)
    flip_samples = np.where(transitions != 0)[0]
    flip_times_harp = harp_pd_time[flip_samples]

    # Find BV times of flips
    Timestamp = frame_events['Timestamp'].values
    Sync = frame_events['Sync'].values
    Trial = frame_events['Trial']
    Frame = frame_events['Frame']
    bv_diff = np.abs(np.diff(Sync))
    flip_times_bv = np.squeeze(Timestamp[np.where(bv_diff==1)[0]])
    # since the first flip is at time zero and not detected by the above method we add it manually:
    flip_times_bv = np.insert(flip_times_bv,0,Timestamp[0])

    # Find TL times when pd flips
    # smooth PD trace
    tl_pd_smoothed = pd.Series(tl_pd).rolling(window=20, min_periods=1).mean().values
    tl_pd_high = np.percentile(tl_pd_smoothed, 99)
    tl_pd_low = np.percentile(tl_pd_smoothed, 1)
    tl_pd_threshold = tl_pd_low + ((tl_pd_high - tl_pd_low)/2)
    tl_pd_thresholded = np.squeeze(tl_pd_smoothed > tl_pd_threshold).astype(int)
    # Set PD signal before first digital flip to value at time of first digital flip
    # tl_pd_thresholded[0:flip_samples_dig_tl[0]] = tl_pd_thresholded[flip_samples_dig_tl[0]]
    tl_pd_thresholded_diff = np.abs(np.diff(tl_pd_thresholded))
    flip_times_pd_tl = np.squeeze(tl_time[np.where(tl_pd_thresholded_diff == 1)])

    # Find lengths of high-low trail start pulses
    tl_pd2_smoothed = pd.Series(tl_pd2).rolling(window=30, min_periods=1).mean().values
    tl_pd2_high = 0.3
    tl_pd2_low = 0.1
    tl_pd2_threshold = tl_pd2_low + ((tl_pd2_high - tl_pd2_low)/2)
    tl_pd2_thresholded = tl_pd2_smoothed
    tl_pd2_thresholded[tl_pd2_thresholded>tl_pd2_high] = 1
    tl_pd2_thresholded[tl_pd2_thresholded<=tl_pd2_low] = -1
    # set tl_pd2_thresholded which are not 1 or -1 to 0
    tl_pd2_thresholded[(tl_pd2_thresholded != 1) & (tl_pd2_thresholded != -1)] = 0
    all_high_pulses = (tl_pd2_thresholded == 1).astype(int)
    all_high_pulse_widths = np.diff(np.where(np.abs(np.diff(all_high_pulses)))[0])
    all_high_pulse_widths = all_high_pulse_widths[all_high_pulse_widths < 5000]
    all_low_pulses = (tl_pd2_thresholded == -1).astype(int)
    all_low_pulse_widths = np.diff(np.where(np.abs(np.diff(all_low_pulses)))[0])
    all_low_pulse_widths = all_low_pulse_widths[all_low_pulse_widths < 5000]
    # plot histogram of pulse widths in 2 subplots
    plt.figure()
    plt.subplot(2,1,1)
    plt.hist(all_high_pulse_widths,bins=np.arange(400,500,5))
    plt.xlim([400,500])
    plt.title('High pulse widths')
    plt.subplot(2,1,2)
    plt.hist(all_low_pulse_widths,bins=np.arange(400,500,5))
    plt.xlim([400,500])
    plt.title('Low pulse widths')
    plt.show()

        
    

    # Find TL times when digital flips
    # REMOVE ONCE BV DIGITAL SIGNAL IS WORKING
    #print('!!!!!!!!!REMOVE ONCE BV DIGITAL SIGNAL IS WORKING!!!!!!!!!')
    #tl_bv_sync = tl_pd_thresholded * 5

    tl_dig_thresholded = np.squeeze((tl_bv_sync > 2.5).astype(int))
    tl_dig_thresholded_diff = np.abs(np.diff(tl_dig_thresholded))
    flip_samples_dig_tl = np.where(tl_dig_thresholded_diff == 1)[0]
    flip_times_dig_tl = np.squeeze(tl_time[np.where(tl_dig_thresholded_diff == 1)])



    # plt.plot(tl_dig_thresholded,color='k')
    # plt.plot(tl_pd_thresholded,color='r')

    # plt.plot(tl_bv_sync,color='k')
    # plt.plot(tl_pd,color='r')    

    # find the moments when the (bonsai) trial number increments
    trialOnsetTimesBV = Timestamp[np.where(np.diff(Trial)==1)[0]]
    # add in first trial onset
    trialOnsetTimesBV = np.insert(trialOnsetTimesBV,0,Timestamp[0]) 


    min_pulses = min(len(flip_times_bv),len(flip_times_pd_tl),len(flip_times_harp))

    print(f'BV pulses           =  : {len(flip_times_bv)}')
    print(f'Harp PD pulses      =  : {len(flip_times_harp)}')
    print(f'TL PD pulses        =  : {len(flip_times_pd_tl)}')
    print(f'TL digital pulses   =  : {len(flip_times_dig_tl)}')

    # TL vs BV drift as a function of TL time
    # TL_BV_pulse_time_diff = (flip_times_dig_tl[0:min_pulses-1]-flip_times_dig_tl[0]) - (flip_times_bv[0:min_pulses-1]-flip_times_bv[0])

    # Plots data with no corrections for lost time btw trials
    # Align all data to time of first pulse
    time_points_spacing = 100
    #plt.plot(tl_time[::time_points_spacing]-flip_times_dig_tl[0],tl_dig_thresholded[::time_points_spacing],label='DIGITAL-DAQ',color='pink')
    # plt.plot(tl_time[::time_points_spacing]-flip_times_pd_tl[0],tl_pd_thresholded[::time_points_spacing],label='Photodiode-DAQ',color='red')
    # plt.plot(Timestamp-flip_times_bv[0],Sync,label='DIGITAL-Bonsai',color='black')
    # plt.plot(harp_pd_time[::time_points_spacing]-flip_times_harp[0],harp_pd_thresholded[::time_points_spacing],label='Photodiode-HARP',color='gray')
    # plt.scatter(trialOnsetTimesBV-flip_times_bv[0],np.ones(trialOnsetTimesBV.shape)*0,color='r',marker='x',label='Trial start (Bonsai)')
    # plt.legend()
    # plt.show()

    # Plot timing pulse times relative to bonsai pulse time
    # min_pulses = np.min([len(flip_times_dig_tl),len(flip_times_bv),len(flip_times_pd_tl),len(flip_times_harp)])
    # plt.plot(flip_times_dig_tl[0:min_pulses]-flip_times_dig_tl[0]-flip_times_bv[0:min_pulses],label='DIGITAL-DAQ',color='pink')
    # plt.plot(flip_times_pd_tl[0:min_pulses]-flip_times_pd_tl[0]-flip_times_bv[0:min_pulses],label='Photodiode-DAQ',color='red')
    # plt.plot(flip_times_bv[0:min_pulses]-flip_times_bv[0]-flip_times_bv[0:min_pulses],label='DIGITAL-Bonsai',color='black')
    # plt.plot(flip_times_harp[0:min_pulses]-flip_times_harp[0]-flip_times_bv[0:min_pulses],label='Photodiode-HARP',color='gray')
    # plt.legend()
    # plt.show()
    # plt.xlabel('Timing pulse number')
    # plt.ylabel('Timing drift (secs) of each aquired\ntiming pulse signal relative to the bonsai "clock"')

    # plot BV sync pulses and BV trial increment times
    # plt.plot(Timestamp,Sync)
    # plt.scatter(trialOnsetTimesBV,np.ones(120)*-0.1,color='r',marker='o')
    # plt.show()
    # //////////////////////////////////////
    # Make model to convert BV time to TL time
    linear_interpolator_bv_2_tl = interp1d(flip_times_bv[:min_pulses], flip_times_pd_tl[:min_pulses], kind='linear', fill_value="extrapolate")
    trialOnsetTimesBV_tl = linear_interpolator_bv_2_tl(trialOnsetTimesBV) # trial onset times in TL timebase
    flip_times_bv_tl = linear_interpolator_bv_2_tl(flip_times_bv)         # flip times in TL timebase
    bv_timestamp_2_tl = linear_interpolator_bv_2_tl(Timestamp)            # BV timestamps in TL timebase
    # model to convert harp time to TL time
    linear_interpolator_harp_2_tl = interp1d(flip_times_harp[:min_pulses], flip_times_pd_tl[:min_pulses], kind='linear', fill_value="extrapolate")
    harp_timestamp_tl = linear_interpolator_harp_2_tl(harp_pd_time)     # harp timestamps in TL timebase
    # //////////////////////////////////////
    # Plotting  

    if plot_on:
        plt.figure()
        #plt.subplot(2,1,1)
        # show waveform of a specific pulse detected in TL and BV
        trial_number = 0 # trial number to use for viewing x position of aligned signals
        pulse_num = 0 # pulse to use for initial alignment tl bv alignment
        use_pd = False
        
        pulse_time_tl = flip_times_pd_tl[pulse_num]
        pulse_time_bv = flip_times_bv[pulse_num]
        tl_relative_time = tl_time - pulse_time_tl
        # to show the BV signal using BV timebase
        # bv_relative_time = Timestamp - pulse_time_bv
        # to show the BV signal using the TL timebase
        bv_relative_time_converted_tl = linear_interpolator_bv_2_tl(Timestamp) - pulse_time_tl
        harp_relative_timestamp_tl  = harp_timestamp_tl - pulse_time_tl

        tl_plot_freq = 500 # Hz
        tl_plot_sample_spacing = int(1000 / tl_plot_freq)
        # Plotting digital signal from TL
        plt.plot(np.squeeze(tl_relative_time[::tl_plot_sample_spacing]), np.squeeze(tl_dig_thresholded[::tl_plot_sample_spacing]),label='BV in TL')
        # Plotting digital signal using the TL timebase
        plt.plot(np.squeeze(bv_relative_time_converted_tl), np.squeeze(Sync)+1.1,label='BV in BV')
        # Plotting TL PD signal
        plt.plot(np.squeeze(tl_relative_time[::tl_plot_sample_spacing]), (np.squeeze(tl_pd_thresholded[::tl_plot_sample_spacing])/3)+2.2,label='PD in TL',color='g')
        plt.plot(np.squeeze(tl_relative_time[::tl_plot_sample_spacing]), (np.squeeze(tl_pd2[::tl_plot_sample_spacing]))+2.2,label='PD in TL',color='black')

        # Plot Harp PD signal
        plt.plot(harp_relative_timestamp_tl[::tl_plot_sample_spacing], (harp_pd_thresholded[::tl_plot_sample_spacing]/3)+2.2,label='PD in Harp',color='y')
        # Plotting all frame events:
        # plt.scatter(bv_timestamp_2_tl-pulse_time_tl,np.ones(Timestamp.shape)*3.3,color='r',marker='.')        
        # Plotting all trial starts
        plt.scatter(trialOnsetTimesBV_tl - pulse_time_tl, np.ones(trialOnsetTimesBV.shape)*-0.1,color='g',marker='o')
        

        window_size = 30 # secs 
        # if use_pd:
        #     plot_start = tl_pd_trial_start_times[trial_number] - (window_size/2) - pulse_time_tl
        #     plt.xlim(plot_start, plot_start+window_size)
        # else:
        #     plot_start = trialOnsetTimesBV[trial_number] - (window_size/2) - pulse_time_bv
        #     plt.xlim(plot_start, plot_start+window_size)

        plt.title(f'BV and TL signals aligned using pulse {pulse_num}\n{expID}')
        plt.legend() 

        # plot PD timed trial starts vs. BV trial number increment timed trial starts
        # plt.subplot(2,1,2)
        # if use_pd:
        #     # compare PD trial start times with BV trial number increment times
        #     drift_per_hour = np.max((tl_pd_trial_start_times-tl_pd_trial_start_times[0])-(trialOnsetTimesBV-trialOnsetTimesBV[0])) / (tl_pd_trial_start_times[-1] - tl_pd_trial_start_times[0]) * 3600
        #     plt.plot((tl_pd_trial_start_times-tl_pd_trial_start_times[0])-(trialOnsetTimesBV-trialOnsetTimesBV[0]),label=f'drift per hour = {drift_per_hour}')
        #     plt.legend()    
        #     plt.title('Relative times of PD hi and BV trial increment derived trial onsets')
        # else:
        #     # compare TL captured sync pulse times with BV captured sync pulse times
        #     flip_count_diff = len(flip_times_dig_tl) - len(flip_times_bv)
        #     min_flips = np.min([np.squeeze(len(flip_times_dig_tl)),np.squeeze(len(flip_times_bv))])
        #     drift_per_hour              = ((flip_times_dig_tl[min_flips-1]-flip_times_dig_tl[0]) - (flip_times_bv[min_flips-1]-flip_times_bv[0])) / (flip_times_dig_tl[-1] - flip_times_dig_tl[0]) * 3600
        #     drift_per_hour_corrected    = ((flip_times_dig_tl[min_flips-1]-flip_times_dig_tl[0]) - (flip_times_bv_tl[min_flips-1]-flip_times_bv_tl[0])) / (flip_times_dig_tl[-1] - flip_times_dig_tl[0]) * 3600

        #     plt.plot((flip_times_dig_tl[0:min_flips-1]-flip_times_dig_tl[0]) - (flip_times_bv[0:min_flips-1]-flip_times_bv[0]),label=f'Raw: drift/hr={drift_per_hour}, flip diff={flip_count_diff}')
        #     # plt.plot((flip_times_tl[0:min_flips-1]-flip_times_tl[0]) - (flip_times_bv_tl[0:min_flips-1]-flip_times_bv_tl[0]),label=f'Corrected: drift/hr={drift_per_hour_corrected}, flip diff={flip_count_diff}')
        #     plt.legend()  
        #     plt.xlabel('Timing pulse number')
        #     plt.ylabel('Time difference between Bonvision\n'
        #                'pulse time and independent DAQ\n'
        #                'pulse time (s)')
        #     plt.title('Relative times of TL recorded BV flips and BV recorded BV flips')

        plt.show(block=True)
    

    #min_len = np.min([np.squeeze(len(flip_times_dig_tl)),np.squeeze(len(flip_times_bv))])
    #initial_time_offset = np.median(flip_times_bv[0:10] - flip_times_dig_tl[0:10])
    #final_time_offset = np.median(flip_times_bv[min_len-10:min_len] - flip_times_dig_tl[min_len-10:min_len])
    #drift = (final_time_offset - initial_time_offset) / (flip_times_dig_tl[min_len-1] - flip_times_dig_tl[0]) * 3600

    # return drift    

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

    allExp = ['2025-01-16_01_YBBT002']
    allExp = ['2025-01-16_01_YBBT002']
    allExp = ['2025-02-25_03_ESPM126']
    for expID in allExp:
        drift = bv_timing_bug2(userID, expID,plot_on=True)
        print(expID + ', ' + str(drift))

if __name__ == "__main__":
    main()