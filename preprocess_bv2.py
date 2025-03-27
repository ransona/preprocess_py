import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy import interpolate
from scipy.io import loadmat
import matplotlib.pyplot as plt
import organise_paths
import pickle
from datetime import datetime
import harp

def run_preprocess_bv2(userID, expID):
    print('Starting run_preprocess_bv...')
    # filter_timing_pulses = True allows removal of timing pulses with duration < min_pulse_width
    # this is used to deal with random fast alterations that bonsai still sometimes produces
    # which can be detected in the electrical signal but not in the photodiode necessarily
    # we therefore remove flips of < min_pulse_width duration
    filter_flips = True
    min_pulse_width = 0.05 # seconds
    max_pulse_width = 0.5 # seconds 

    animalID, remote_repository_root, \
    processed_root, exp_dir_processed, \
        exp_dir_raw = organise_paths.find_paths(userID, expID)
    exp_dir_processed_recordings = os.path.join(processed_root, animalID, expID,'recordings')
    # load the stimulus parameter file produced by matlab by the bGUI
    # this includes stim parameters and stimulus order
    try:
        stim_params = loadmat(os.path.join(exp_dir_raw, expID + '_stim.mat'))
    except:
        raise Exception('Stimulus parameter file not found - this experiment was probably from pre-Dec 2021.')
    
    # load timeline
    Timeline = loadmat(os.path.join(exp_dir_raw, expID + '_Timeline.mat'))
    Timeline = Timeline['timelineSession']
    # get timeline file in a usable format after importing to python
    tl_chNames = Timeline['chNames'][0][0][0][0:]
    tl_daqData = Timeline['daqData'][0,0]
    tl_time    = Timeline['time'][0][0]
    tl_time    = np.squeeze(tl_time)
    # load BV data
    frame_events = pd.read_csv(os.path.join(exp_dir_raw, expID + '_FrameEvents.csv'), names=['Frame', 'Timestamp', 'Sync', 'Trial'],
                            header=None, skiprows=[0], dtype={'Frame':np.float32, 'Timestamp':np.float32, 'Sync':np.float32, 'Trial':np.float32})
    bv_encoder = pd.read_csv(os.path.join(exp_dir_raw, expID + '_Encoder.csv'), names=['Frame', 'Timestamp', 'Unknown', 'Encoder'],
                            header=None, skiprows=[0], dtype={'Frame':np.float32, 'Timestamp':np.float32, 'Sync':np.float32, 'Trial':np.float32})    

    # load Harp raw data
    harp_data_path = os.path.join(exp_dir_raw, expID + '_Behavior_Event44.bin')
    data_read = harp.io.read(harp_data_path)
    data_read_np = np.array(data_read)
    # data_read_np[0:400,0] = data_read_np[400,0]
    harp_pd = data_read_np[:,0]
    harp_encoder = data_read_np[:,1]
    harp_time = np.arange(0, len(harp_pd)/1000, 1/1000)

    # /////////////// DETECTING TIMING PULSES ///////////////

    # Find Harp times when PD flip
    #  Threshold the Harp PD signal and detect flips
    harp_pd_smoothed = pd.Series(harp_pd).rolling(window=20, min_periods=1).mean().values
    harp_pd_high = np.percentile(harp_pd_smoothed, 99)
    harp_pd_low = np.percentile(harp_pd_smoothed, 1)   
    harp_pd_on_off_ratio = (harp_pd_high - harp_pd_low) / harp_pd_low 
    if harp_pd_on_off_ratio > 10:
        harp_pd_valid = True  
    else:
        harp_pd_valid = False   

    harp_pd_threshold = harp_pd_low + ((harp_pd_high - harp_pd_low)*0.5)
    harp_pd_thresholded = np.where(harp_pd_smoothed < harp_pd_threshold, 0, 1)
    transitions = np.diff(harp_pd_thresholded)
    flip_samples = np.where(transitions != 0)[0]
    flip_times_harp = harp_time[flip_samples]
    
    # Find BV times when digital flips
    Timestamp = frame_events['Timestamp'].values
    Sync = frame_events['Sync'].values
    Trial = frame_events['Trial']
    Frame = frame_events['Frame']
    bv_diff = np.abs(np.diff(Sync))
    flip_times_bv_bv = np.squeeze(Timestamp[np.where(bv_diff==1)[0]])
    # since the first flip is at time zero and thus not detectable we add it manually:
    flip_times_bv_bv = np.insert(flip_times_bv_bv,0,Timestamp[0])

    # Find TL times when PD flips
    tl_pd_ch = np.where(np.isin(tl_chNames, 'Photodiode'))
    tl_pd = np.squeeze(tl_daqData[:, tl_pd_ch])
    # Needs to be smoothed because the photodiode signal is noisy and monitor blanking can further screw it up
    # There should also be a RC filter in the photodiode signal path to smooth high frequency noise
    tl_pd_smoothed = pd.Series(tl_pd).rolling(window=20, min_periods=1).mean().values
    # Calculate threshold for PD signal
    tl_pd_high = np.percentile(tl_pd_smoothed, 99)
    tl_pd_low = np.percentile(tl_pd_smoothed, 1)
    tl_pd_on_off_ratio = (tl_pd_high - tl_pd_low) / tl_pd_low

    if tl_pd_on_off_ratio > 10:
        tl_pd_valid = True
    else:
        tl_pd_valid = False

    tl_pd_threshold = tl_pd_low + ((tl_pd_high - tl_pd_low)*0.5)
    tl_pd_thresholded = np.squeeze(tl_pd_smoothed > tl_pd_threshold).astype(int)
    # Detect both rising and falling edges
    tl_pd_thresholded_diff = np.abs(np.diff(tl_pd_thresholded))
    flip_times_pd_tl = np.squeeze(tl_time[np.where(tl_pd_thresholded_diff == 1)])

    # Find TL times when BV Digital flips
    tl_bv_ch = np.where(np.isin(tl_chNames, 'Bonvision'))
    tl_bv = np.squeeze(tl_daqData[:, tl_bv_ch])
    # Needs to be smoothed because the photodiode signal is noisy and monitor blanking can further screw it up
    # There should also be a RC filter in the photodiode signal path to smooth high frequency noise
    tl_bv_smoothed = pd.Series(tl_bv).rolling(window=1, min_periods=1).mean().values
    # Calculate threshold for PD signal
    tl_bv_high = np.percentile(tl_bv_smoothed, 99)
    tl_bv_low = np.percentile(tl_bv_smoothed, 1)
    tl_bv_threshold = tl_bv_low + ((tl_bv_high - tl_bv_low)/2)
    tl_bv_thresholded = np.squeeze(tl_bv_smoothed > tl_bv_threshold).astype(int)
    # Detect both rising and falling edges
    tl_bv_thresholded_diff = np.abs(np.diff(tl_bv_thresholded))
    flip_times_bv_tl = np.squeeze(tl_time[np.where(tl_bv_thresholded_diff == 1)])

    # in experiments with screens off there is no PD signal and thus no flips and 
    # can not therefore convert from TL time to Harp time. In these cases the
    # the encoder data is not retrievable from the harp data file in the usual way
    # and we thus use the BV encoder data
    if tl_pd_valid == True and harp_pd_valid == True:
        print('Both TL and Harp PD signals are valid')
        pd_valid = True
    else:
        print('*** Warning: One or both of the PD signals are invalid. If this is an experiment with screens off this is expected. If not, there is a problem. ***')
        choice = input("Do you want to continue? (y/n): ").strip().lower()
        
        if choice != 'y':
            print("Exiting...")
            return
        
        pd_valid = False

    if filter_flips:
        min_pulses_unfiltered = min(len(flip_times_bv_bv),len(flip_times_pd_tl),len(flip_times_harp))
        # before filtering check status
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Before filtering timing signal:')
        if len({len(flip_times_pd_tl),len(flip_times_harp),len(flip_times_bv_tl),len(flip_times_bv_bv)}) != 1:
            print('Number of flips detected in TL, BV and Harp do not match:')
            print('TL PD flips = ' + str(len(flip_times_pd_tl)))
            print('BV TL flips = ' + str(len(flip_times_bv_tl)))
            print('BV flips = ' + str(len(flip_times_bv_bv)))
            print('Harp flips = ' + str(len(flip_times_harp)))
        
        pulse_time_diff_tl_bv_unfiltered = (flip_times_pd_tl[0:min_pulses_unfiltered]-flip_times_pd_tl[0])-(flip_times_bv_tl[0:min_pulses_unfiltered]-flip_times_bv_tl[0])
        # remove all pulses of < a certain width in tl/harp time

        # all_diff = np.diff(flip_times_pd_tl)
        # all_diff = all_diff[all_diff < 1]
        flip_times_pd_tl_filtered = flip_times_pd_tl[np.where((np.diff(flip_times_pd_tl) > min_pulse_width) & (np.diff(flip_times_pd_tl) < max_pulse_width))[0]]
        flip_times_harp_filtered = flip_times_harp[np.where((np.diff(flip_times_harp) > min_pulse_width) & (np.diff(flip_times_harp) < max_pulse_width))[0]]
        flip_times_dig_tl_filtered = flip_times_bv_tl[np.where((np.diff(flip_times_bv_tl) > min_pulse_width) & (np.diff(flip_times_bv_tl) < max_pulse_width))[0]]
        flips_to_keep_bv = np.where((np.diff(flip_times_bv_tl) > min_pulse_width) & (np.diff(flip_times_bv_tl) < max_pulse_width))[0]

        # flip_times_harp_filtered = flip_times_harp[np.where(np.diff(flip_times_harp) > min_width)[0]]
        # flip_times_dig_tl_filtered = flip_times_dig_tl[np.where(np.diff(flip_times_dig_tl) > min_width)[0]]
        # flips_to_keep_bv = np.where(np.diff(flip_times_dig_tl) > min_width)[0]
        
        flip_times_bv_bv = flip_times_bv_bv[flips_to_keep_bv]
        flip_times_harp = flip_times_harp_filtered
        flip_times_pd_tl = flip_times_pd_tl_filtered
        flip_times_bv_tl = flip_times_dig_tl_filtered

        # do sanity checks
        print('')
        print('After filtering timing signal:')
        if pd_valid:
            # number of flips should be the same on all systems if PD is valid
            if (len({len(flip_times_harp),len(flip_times_bv_bv)}) != 1) and (len({len(flip_times_pd_tl),len(flip_times_bv_tl),len(flip_times_bv_bv)}) == 1):
                # harp flip count is wrong but others are rigth so can still use BV data for encoder
                print('Number of flips detected in Harp and BV do not match after filtering both other timing pulses do match. You may thus continue by using rotary encoder data from BV log instead of harp:')
                print('Harp flips = ' + str(len(flip_times_harp)))
                print('BV flips = ' + str(len(flip_times_bv_bv)))
                print('This issue should not occur on data acquired after 25/03/2025 - please contact AR if you see this issue on data after this date.')
                choice = input("Do you want to continue? (y/n): ").strip().lower()
                if choice != 'y':
                    print("Exiting...")
                    return
                # else set to not use PD and thus not use Harp for encoder
                pd_valid = False                
            elif len({len(flip_times_pd_tl),len(flip_times_harp),len(flip_times_bv_tl),len(flip_times_bv_bv)}) != 1:
                print('Number of flips detected in TL, BV and Harp do not match:')
                print('TL PD flips = ' + str(len(flip_times_pd_tl)))
                print('BV TL flips = ' + str(len(flip_times_bv_tl)))
                print('BV flips = ' + str(len(flip_times_bv_bv)))
                print('Harp flips = ' + str(len(flip_times_harp)))
                raise ValueError('Pulse count mismatch')
            else:
                print('Number of flips detected in TL, BV and Harp match:')
                print('BV flips = ' + str(len(flip_times_bv_tl)))
            # the relative times of flips should be near identical between flip_times_pd_tl and flip_times_bv_tl
            pd_tl_v_bv_tl_jitter = np.abs((flip_times_pd_tl-flip_times_pd_tl[0]) - (flip_times_bv_tl-flip_times_bv_tl[0]))
            if max(pd_tl_v_bv_tl_jitter) > 50:
                print('Jitter between TL and BV timing pulses is too large:')
                print('Max jitter = ' + str(round(max(pd_tl_v_bv_tl_jitter)*1000)) + ' ms')
                raise ValueError('Jitter mismatch')
            else:
                print('Jitter between TL and BV timing pulses is acceptable:')
                print('Median jitter = ' + str(round(np.median(pd_tl_v_bv_tl_jitter)*1000))+ ' ms')
                print('Max jitter = ' + str(round(max(pd_tl_v_bv_tl_jitter)*1000)) + ' ms')
        else:
            # number of flips should be the same on BV and TL
            if len(flip_times_bv_bv) != len(flip_times_bv_tl):
                print('Number of flips detected in BV and TL do not match:')
                print('BV flips = ' + str(len(flip_times_bv_bv)))
                print('TL flips = ' + str(len(flip_times_bv_tl)))
                raise ValueError('Pulse count mismatch')
            else:
                print('Number of flips detected in BV and TL match:')
                print('BV flips = ' + str(len(flip_times_bv_bv)))
        print('Filtering complete')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')


    # Use the BV ground truth for the number of flips that should be present
    true_flips = len(flip_times_bv_bv)

    # Check that the number of flips detected in the PD signal is the same as the number of flips detected in the BV signal

    # Fit model to convert BV time to TL time either using PD or digital flip signal from BV
    if pd_valid:
        # use PD
        linear_interpolator_bv_2_tl = interp1d(flip_times_bv_bv[0:true_flips], flip_times_pd_tl[0:true_flips], kind='linear', fill_value="extrapolate")
    else:
        # use digital signal
        linear_interpolator_bv_2_tl = interp1d(flip_times_bv_bv[0:true_flips], flip_times_bv_tl[0:true_flips], kind='linear', fill_value="extrapolate")
                                               
    if pd_valid:
        # Fit model to convert Harp time to TL time
        linear_interpolator_harp_2_tl = interp1d(flip_times_harp[0:true_flips], flip_times_pd_tl[0:true_flips], kind='linear', fill_value="extrapolate")

        # Check all systems registered the same number of pulses
        if len(flip_times_harp) == len(flip_times_bv_bv) == len(flip_times_pd_tl):
            print ('Pulse count matches accross TL/BV/Harp')
            print ('Pulse count = ' + str(len(flip_times_harp)))
        else:
            print('Harp pulses = ' + str(len(flip_times_harp)))
            print('BV pulses = ' + str(len(flip_times_bv_bv)))
            print('TL pulses = ' + str(len(flip_times_pd_tl)))
            raise ValueError('Pulse count mismatch')
    else:
        # Check all systems registered the same number of pulses
        if len(flip_times_bv_bv) == len(flip_times_bv_tl):
            print ('Pulse count matches accross TL/BV')
            print ('Pulse count = ' + str(len(flip_times_bv_bv)))
        else:
            print('BV pulses = ' + str(len(flip_times_bv_bv)))
            print('TL pulses = ' + str(len(flip_times_bv_tl)))
            raise ValueError('Pulse count mismatch')        
    
    # get trial onset times
    # in BV time
    # find the moments when the (bonsai) trial number increments
    trialOnsetTimesBV = Timestamp[np.where(np.diff(Trial)==1)]
    # add in first trial onset
    trialOnsetTimesBV = np.insert(trialOnsetTimesBV,0,Timestamp[0])
    # in TL time
    trialOnsetTimesBV_tl = linear_interpolator_bv_2_tl(trialOnsetTimesBV)

    # load matlab expData file
    expData = loadmat(os.path.join(exp_dir_raw, expID + '_stim.mat'))
    stims = expData['expDat']['stims']
    stims = stims[0][0][0]

    stim_info = pd.read_csv(os.path.join(exp_dir_raw, expID + '_stim.csv'))
    stim_order = pd.read_csv(os.path.join(exp_dir_raw, expID + '_stim_order.csv'), header=None)

    # make a matrix for csv output of trial onset time and trial stimulus type
    # check number of trial onsets matches between bonvision and bGUI
    if len(trialOnsetTimesBV_tl) != stim_order.shape[0]:
        raise ValueError(
            'Number of trial onsets doesn\'t match between bonvision and bGUI - there is a likely logging issue')
    else:
        print('Number of trial onsets matches between bonvision and bGUI')
        print('Number of trials = ' + str(len(trialOnsetTimesBV_tl)))
    # make the matrix of trial onset times
    trialTimeMatrix = np.column_stack((trialOnsetTimesBV_tl, stim_order.values))

    # Add running trace
    if pd_valid:
        # then we can use harp for encoder
        wheel_pos = harp_encoder
        wheel_timestamps = linear_interpolator_harp_2_tl(harp_time)
    else:
        # we use BV for encoder
        wheel_pos = bv_encoder['Encoder'].values
        wheel_timestamps = linear_interpolator_bv_2_tl(bv_encoder['Timestamp'].values)

    # deal with wrap around of rotary encoder position
    wheel_pos_dif = np.diff(wheel_pos)
    wheel_pos_dif[wheel_pos_dif > 50000] -= 2**16
    wheel_pos_dif[wheel_pos_dif < -50000] += 2**16
    wheel_pos = np.cumsum(wheel_pos_dif)
    wheel_pos = np.append(wheel_pos,wheel_pos[-1])
    
    # Resample wheel to 20Hz
    resample_freq = 20
    wheel_linear_timescale = np.arange(0, np.floor(wheel_timestamps[-1]), 1/resample_freq)
    # Create the interpolater for resampling
    wheel_resampler = interpolate.interp1d(wheel_timestamps, wheel_pos, kind='linear',fill_value=(wheel_pos[0], wheel_pos[-1]), bounds_error=False)
    # Infer the wheel pos at each point on linear timescale 
    wheel_pos_resampled = wheel_resampler(wheel_linear_timescale)
    # smooth this position data to deal with the rotary encoder encoding discrete steps
    smooth_window = 10 # window size for smoothing (10 = 0.5 secs)
    wheel_pos_smooth = np.convolve(wheel_pos_resampled, np.ones(smooth_window)/smooth_window, mode='same')
    # set smooth_window at start and end to the first and last value of the unsmoothed data
    wheel_pos_smooth[0:smooth_window] = wheel_pos_resampled[0]
    wheel_pos_smooth[-smooth_window:] = wheel_pos_resampled[-1]
    # Calc diff between position samples and then mutiply by circumference of the running wheel
    wheel_diameter = 17.5 # cm
    encoder_resolution = 1024
    wheel_circumference = wheel_diameter * np.pi
    # mouse velocity in cm/sample (at 20Hz)
    wheel_velocity = np.diff(wheel_pos_smooth) * (wheel_circumference / encoder_resolution)
    wheel_velocity = np.append(wheel_velocity, wheel_velocity[-1])
    # mouse velocity in cm/s
    wheel_velocity = wheel_velocity * resample_freq
    # Save data
    wheel = {}
    wheel['position'] = np.array(wheel_pos_resampled)
    wheel['position_smoothed'] = np.array(wheel_pos_smooth)
    wheel['speed'] = np.array(wheel_velocity)
    wheel['t'] = np.array(wheel_linear_timescale)
    with open(os.path.join(exp_dir_processed_recordings, 'wheel.pickle'), 'wb') as f: pickle.dump(wheel, f)

    # output a csv file which contains dataframe of all trials with first column showing trial onset time
    # read the all trials file, append trial onset times to first column (trialOnsetTimesTL)
    all_trials = pd.read_csv(os.path.join(exp_dir_raw, expID + '_all_trials.csv'))
    all_trials.insert(0,'time',trialOnsetTimesBV_tl)
    all_trials.to_csv(os.path.join(exp_dir_processed, expID + '_all_trials.csv'), index=False)
    print('Done without errors')

    # for debugging:
def main():
    # userID = 'melinatimplalexi'
    userID = 'pmateosaparicio'
    #userID = 'adamranson'
    expID = '2025-03-13_02_ESPM126'
    run_preprocess_bv2(userID, expID)

if __name__ == "__main__":
    main()