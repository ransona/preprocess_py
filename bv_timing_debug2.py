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

def bv_timing_bug2(userID, expID,plot_on,filter_on=False):
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
    harp_encoder = data_read_np[:,1]
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
    bv_encoder = pd.read_csv(os.path.join(exp_dir_raw, expID + '_Encoder.csv'), names=['Frame', 'Timestamp', 'Unknown', 'Encoder'],
                            header=None, skiprows=[0], dtype={'Frame':np.float32, 'Timestamp':np.float32, 'Unknown':np.float32, 'Encoder':np.float32})
    # plt.figure()
    # plt.plot(harp_pd_time+13.35,harp_pd/max(harp_pd),label='Harp PD',color='r')
    # plt.plot(tl_time,tl_pd/max(tl_pd),label='TL PD',color='b')
    # plt.plot(tl_time+0.07,tl_bv_sync/max(tl_bv_sync),label='TL BV',color='k')
    # plt.show(block=False) 

    # /////////////// DETECTING TIMING PULSES ///////////////
    PD_smoothing_window = 20
    # Find Harp times when PD flip
    #  Threshold the Harp PD signal and detect flips
    harp_pd_smoothed = pd.Series(harp_pd).rolling(window=PD_smoothing_window, min_periods=1).mean().values
    harp_pd_high = np.percentile(harp_pd_smoothed, 99)
    harp_pd_low = np.percentile(harp_pd_smoothed, 1)  
    harp_pd_on_off_ratio = (harp_pd_high - harp_pd_low) / harp_pd_low 
    harp_pd_threshold = harp_pd_low + ((harp_pd_high - harp_pd_low)*0.5)
     
    harp_pd_thresholded = np.where(harp_pd_smoothed < harp_pd_threshold, 0, 1)
    transitions = np.diff(harp_pd_thresholded)
    flip_samples = np.where(transitions != 0)[0]
    flip_times_harp = harp_pd_time[flip_samples]

    # Find BV times of flips
    Timestamp = frame_events['Timestamp'].values
    Sync = frame_events['Sync'].values
    Encoder = bv_encoder['Encoder'].values
    Trial = frame_events['Trial']
    Frame = frame_events['Frame']
    bv_diff = np.abs(np.diff(Sync))
    flip_times_bv = np.squeeze(Timestamp[np.where(bv_diff==1)[0]])
    # since the first flip is at time zero and not detected by the above method we add it manually:
    flip_times_bv = np.insert(flip_times_bv,0,Timestamp[0])

    # Find TL times when pd flips
    # smooth PD trace
    tl_pd_smoothed = pd.Series(tl_pd).rolling(window=PD_smoothing_window, min_periods=1).mean().values
    tl_pd_high = np.percentile(tl_pd_smoothed, 99)
    tl_pd_low = np.percentile(tl_pd_smoothed, 1)
    # check high vs low is > a certain percentage change
    tl_pd_on_off_ratio = (tl_pd_high - tl_pd_low) / tl_pd_low

    tl_pd_threshold = tl_pd_low + ((tl_pd_high - tl_pd_low)*0.5)
    tl_pd_thresholded = np.squeeze(tl_pd_smoothed > tl_pd_threshold).astype(int)
    # Set PD signal before first digital flip to value at time of first digital flip
    # tl_pd_thresholded[0:flip_samples_dig_tl[0]] = tl_pd_thresholded[flip_samples_dig_tl[0]]
    tl_pd_thresholded_diff = np.abs(np.diff(tl_pd_thresholded))
    flip_times_pd_tl = np.squeeze(tl_time[np.where(tl_pd_thresholded_diff == 1)])

    # Find lengths of high-low trail start pulses
    tl_pd2_smoothed = pd.Series(tl_pd2).rolling(window=PD_smoothing_window, min_periods=1).mean().values
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
    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.hist(all_high_pulse_widths,bins=np.arange(400,500,5))
    # plt.xlim([400,500])
    # plt.title('High pulse widths')
    # plt.subplot(2,1,2)
    # plt.hist(all_low_pulse_widths,bins=np.arange(400,500,5))
    # plt.xlim([400,500])
    # plt.title('Low pulse widths')
    # plt.show()

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

    # plot harp vs tl timing pulse times relative to first pulse
    # min_pulses = min(len(flip_times_bv),len(flip_times_pd_tl),len(flip_times_harp))
    # plt.figure()
    # plt.plot((flip_times_pd_tl[0:min_pulses]-flip_times_pd_tl[0])-(flip_times_harp[0:min_pulses]-flip_times_harp[0]),label='TL PD')
    # plt.show(block=False)

    # print min diff btw flip_times_dig_tl timing pulses
    min_diff_dig_tl = np.min(np.diff(flip_times_dig_tl))
    print(f'Min diff between TL digital timing pulses: {min_diff_dig_tl}')


    if filter_on:
        print('Pre filtering:')
        print(f'BV pulses           =  : {len(flip_times_bv)}')
        print(f'Harp PD pulses      =  : {len(flip_times_harp)}')
        print(f'TL PD pulses        =  : {len(flip_times_pd_tl)}')
        print(f'TL digital pulses   =  : {len(flip_times_dig_tl)}')        
        # min_pulses_unfiltered = min(len(flip_times_bv),len(flip_times_pd_tl),len(flip_times_harp))
        # pulse_time_diff_tl_bv_unfiltered = (flip_times_pd_tl[0:min_pulses]-flip_times_pd_tl[0])-(flip_times_dig_tl[0:min_pulses]-flip_times_dig_tl[0])
        # remove all pulses of < a certain width in tl/harp time
        min_width = 0.05
        max_width = 0.5
        # all_diff = np.diff(flip_times_pd_tl)
        # all_diff = all_diff[all_diff < 1]
        flip_times_pd_tl_filtered = flip_times_pd_tl[np.where((np.diff(flip_times_pd_tl) > min_width) & (np.diff(flip_times_pd_tl) < max_width))[0]]
        flip_times_harp_filtered = flip_times_harp[np.where((np.diff(flip_times_harp) > min_width) & (np.diff(flip_times_harp) < max_width))[0]]
        flip_times_dig_tl_filtered = flip_times_dig_tl[np.where((np.diff(flip_times_dig_tl) > min_width) & (np.diff(flip_times_dig_tl) < max_width))[0]]
        flips_to_keep_bv = np.where((np.diff(flip_times_dig_tl) > min_width) & (np.diff(flip_times_dig_tl) < max_width))[0]

        # flip_times_harp_filtered = flip_times_harp[np.where(np.diff(flip_times_harp) > min_width)[0]]
        # flip_times_dig_tl_filtered = flip_times_dig_tl[np.where(np.diff(flip_times_dig_tl) > min_width)[0]]
        # flips_to_keep_bv = np.where(np.diff(flip_times_dig_tl) > min_width)[0]
        
        flip_times_bv = flip_times_bv[flips_to_keep_bv]
        flip_times_harp = flip_times_harp_filtered
        flip_times_pd_tl = flip_times_pd_tl_filtered
        flip_times_dig_tl = flip_times_dig_tl_filtered

        # do sanity check to see if stochastic variation in pulse width is correlated between signals
        print('Post filtering:')

    min_pulses = min(len(flip_times_bv),len(flip_times_pd_tl),len(flip_times_harp))

    print(f'BV pulses           =  : {len(flip_times_bv)}')
    print(f'Harp PD pulses      =  : {len(flip_times_harp)}')
    print(f'TL PD pulses        =  : {len(flip_times_pd_tl)}')
    print(f'TL digital pulses   =  : {len(flip_times_dig_tl)}')
    # plot TL PD and TL Digital signals aligned to first flip

    
    # plot harp and TL PD signals aligned to first flip
    first_sample_tl = np.where(tl_time>flip_times_pd_tl[0])[0][0]
    first_sample_harp = np.where(harp_pd_time>flip_times_harp[0])[0][0]
    first_sample_tl_dig = np.where(tl_time>flip_times_dig_tl[0])[0][0]
    plt.figure()
    spacing = 1
    # time axis is samples
    samples_to_plot = np.arange(first_sample_tl,len(tl_pd_smoothed),spacing)
    plt.plot((samples_to_plot-samples_to_plot[0]),tl_pd_smoothed[samples_to_plot]/np.max(tl_pd_smoothed),label='TL PD')
    plt.plot((samples_to_plot-samples_to_plot[0]),tl_dig_thresholded[samples_to_plot]/np.max(tl_pd_smoothed),label='TL BV')
    samples_to_plot = np.arange(first_sample_harp,len(harp_pd_smoothed),spacing)
    plt.plot((samples_to_plot-samples_to_plot[0]),harp_pd_smoothed[samples_to_plot]/np.max(harp_pd_smoothed),label='Harp PD')
    # over this plot the pulse time differences
    pulse_time_diff_tl_harp = (flip_times_pd_tl[0:min_pulses]-flip_times_pd_tl[0])-(flip_times_harp[0:min_pulses]-flip_times_harp[0])
    pulse_time_diff_tl_bv = (flip_times_pd_tl[0:min_pulses]-flip_times_pd_tl[0])-(flip_times_dig_tl[0:min_pulses]-flip_times_dig_tl[0])
    # plt.plot((flip_times_pd_tl[0:min_pulses]-flip_times_pd_tl[0])*1000,pulse_time_diff_tl_harp,label='TL PD - Harp')
    # plt.plot((flip_times_pd_tl[0:min_pulses]-flip_times_pd_tl[0])*1000,pulse_time_diff_tl_bv,label='TL PD - BV Electric')
    # plt.plot((flip_times_pd_tl[0:min_pulses_unfiltered]-flip_times_pd_tl[0])*1000,pulse_time_diff_tl_bv_unfiltered,label='TL PD - BV Electric - unfiltered')
    
    plt.plot((flip_times_pd_tl[0:min_pulses]-flip_times_pd_tl[0])*1000,pulse_time_diff_tl_bv,label='TL PD - BV Electric')
    plt.legend()
    plt.show()

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

    # Detect harp flips which occur after the last TL flip in TL time
    # last valid flip time in TL time
    harp_last_valid_flip_time = flip_times_pd_tl[-1]
    # detect how many flips are after this time
    flip_times_harp_to_tl = linear_interpolator_harp_2_tl(flip_times_harp)
    harp_last_flip_invalid_flips = np.where(flip_times_harp_to_tl > harp_last_valid_flip_time)[0]
    print(f'Number of harp flips after last valid TL flip: {len(harp_last_flip_invalid_flips)}')

    # Plotting  
    plt.figure()
    plt.plot(bv_timestamp_2_tl,Encoder)
    plt.plot(harp_timestamp_tl,harp_encoder)
    plt.legend(['BV', 'Harp'])
    plt.show()
    

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
    userID = 'adamranson'
    #allExp = ['2025-03-12_01_ESPM126'] # cinematic_3
    #allExp = ['2025-03-13_02_ESPM126'] # has 6 extra harp flips
    #allExp = ['2025-03-21_02_TEST'] # new setup 1 hr recording
    allExp = ['2025-03-26_01_ESPM126'] # has a fast BV flip that is missed in PD
    #allExp = ['2025-03-26_02_ESPM126'] # has no issues
    #allExp = ['2025-03-05_02_ESMT204'] # stim artifact


    for expID in allExp:
        drift = bv_timing_bug2(userID, expID,plot_on=True,filter_on=False)
        print(expID + ', ' + str(drift))

if __name__ == "__main__":
    main()