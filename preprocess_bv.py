import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy import interpolate
from scipy.io import loadmat
import matplotlib.pyplot as plt
import organise_paths
import pickle

def run_preprocess_bv(userID, expID):
    print('Starting run_preprocess_bv...')
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

    frame_events = pd.read_csv(os.path.join(exp_dir_raw, expID + '_FrameEvents.csv'), names=['Frame', 'Timestamp', 'Sync', 'Trial'],
                            header=None, skiprows=[0], dtype={'Frame':np.float32, 'Timestamp':np.float32, 'Sync':np.float32, 'Trial':np.float32})

    # Find BV times when digital flips
    Timestamp = frame_events['Timestamp'].values
    Sync = frame_events['Sync'].values
    Trial = frame_events['Trial']
    flip_idx = np.where(np.diff(Sync) == -1)
    flip_times_bv = np.squeeze(Timestamp[np.where((np.diff(Sync) == -1))[0]])

    # Find TL times when digital flips
    bv_ch = np.where(np.isin(tl_chNames, 'Bonvision'))
    tl_dig_thresholded = np.squeeze((tl_daqData[:, bv_ch] > 2.5).astype(int))

    flip_times_tl = np.squeeze(tl_time[0,np.where(np.diff(tl_dig_thresholded) == -1)])
    # Check NI DAQ caught as many sync pulses as BV produced
    pulse_diff = len(flip_times_tl) - len(flip_times_bv)
    print(str(len(flip_times_tl)) + ' pulses found in TL')

    if pulse_diff > 0:
        print(str(pulse_diff) + ' more pulses in TL')
        flip_times_tl = flip_times_tl[:len(flip_times_bv)]
    elif pulse_diff < -1:
        print(str(pulse_diff * -1) + ' more pulses in BV')
        raise ValueError('Pulse mismatch')
    elif pulse_diff == -1:
        print(str(pulse_diff * -1) + ' more pulses in BV')
        print('This issue needs to be monitored to find out why it is happening')
        # could be that previous experiment has not been stopped properly and trial is still running?
        print('Here we work around it...')
        flip_times_bv = flip_times_bv[:len(flip_times_tl)]
    else:
        print('Pulse match')

    # Make model to convert BV time to TL time
    mdl1 = LinearRegression()
    mdl1.fit(flip_times_bv.reshape((-1, 1)), flip_times_tl)

    # get trial onset times
    # in BV time
    # find the moments when the (bonsai) trial number increments
    trialOnsetTimesBV = Timestamp[np.where(np.diff(Trial)==1)]
    # add in first trial onset
    trialOnsetTimesBV = np.insert(trialOnsetTimesBV,0,Timestamp[0])
    # in TL time
    trialOnsetTimesTL = mdl1.predict(pd.DataFrame(trialOnsetTimesBV))

    # load matlab expData file
    expData = loadmat(os.path.join(exp_dir_raw, expID + '_stim.mat'))
    stims = expData['expDat']['stims']
    stims = stims[0][0][0]

    stim_info = pd.read_csv(os.path.join(exp_dir_raw, expID + '_stim.csv'))
    stim_order = pd.read_csv(os.path.join(exp_dir_raw, expID + '_stim_order.csv'), header=None)

    # make a matrix for csv output of trial onset time and trial stimulus type
    # check number of trial onsets matches between bonvision and bGUI
    if len(trialOnsetTimesTL) != stim_order.shape[0]:
        raise ValueError(
            'Number of trial onsets doesn\'t match between bonvision and bGUI - there is a likely logging issue')

    # make the matrix of trial onset times
    trialTimeMatrix = np.column_stack((trialOnsetTimesTL, stim_order.values))

    # Add running trace
    Encoder = pd.read_csv(os.path.join(exp_dir_raw, expID + '_Encoder.csv'), names=['Frame', 'Timestamp', 'Trial', 'Position'],
                            header=None, skiprows=[0]) #, dtype={'Frame':np.float32, 'Timestamp':np.float32, 'Trial':np.int64, 'Position':np.int64})
    wheelPos = Encoder.Position.values
    # deal with wrap around of rotary encoder position
    wheelPosDif = np.diff(wheelPos)
    wheelPosDif[wheelPosDif > 50000] -= 2**16
    wheelPosDif[wheelPosDif < -50000] += 2**16
    wheelPos = np.cumsum(wheelPosDif)
    wheelPos=np.append(wheelPos,wheelPos[-1])
    wheelTimestamps = mdl1.predict(Encoder.Timestamp.values.reshape(-1,1))
    # Resample wheel to linear timescale
    wheelLinearTimescale = np.arange(np.ceil(wheelTimestamps[0]), np.floor(wheelTimestamps[-1]), 0.05)
    # Create the interpolater
    f = interpolate.interp1d(wheelTimestamps, wheelPos, kind='linear')
    # Infer the wheel pos at each point on linear timescale and smooth this position data
    wheelPos2 = pd.Series(f(wheelLinearTimescale)).rolling(window=50, center=True).mean().fillna(method='ffill').fillna(method='bfill')
    # Calc diff between position samples and then mutiply by circumference of the running wheel
    wheelSpeed = ((np.insert(np.diff(wheelPos2.values) * -1, 0, 0) * (62 / 1024)) * 100) # should be multiplied by 20?
    # add filler data for period before bv starts when wheel isn't stored
    filler_t = np.arange(0,wheelLinearTimescale[0],0.05)
    filler_position = np.tile(wheelPos2[0],[filler_t.shape[0]])
    filler_speed = np.tile(wheelSpeed[0],[filler_t.shape[0]])
    # for debugging
    # fig, axs = plt.subplots(5,1, sharex=True)
    # axs[0].plot(wheelTimestamps,Encoder.Position.values)
    # axs[0].set_title('Raw')
    # axs[1].plot(wheelTimestamps[0:-1],wheelPosDif)
    # axs[1].set_title('WheelPosDif')
    # axs[2].plot(wheelTimestamps,wheelPos)
    # axs[2].set_title('WheelPos (wraparound removed)')
    # axs[3].plot(wheelLinearTimescale,wheelPos2)
    # axs[3].set_title('wheelPos2')
    # axs[4].plot(wheelLinearTimescale,wheelSpeed)
    # axs[4].set_title('wheelSpeed')  
    # plt.tight_layout()
    # plt.show() 

    # Save data
    wheel = {}
    wheel['position'] = np.concatenate([filler_position,np.array(wheelPos2)],axis = 0)
    wheel['speed'] = np.concatenate([filler_speed,np.array(wheelSpeed)],axis = 0)
    wheel['t'] = np.concatenate([filler_t,np.array(wheelLinearTimescale)],axis = 0)
    with open(os.path.join(exp_dir_processed_recordings, 'wheel.pickle'), 'wb') as f: pickle.dump(wheel, f)

    # output a csv file which contains dataframe of all trials with first column showing trial onset time
    # read the all trials file, append trial onset times to first column (trialOnsetTimesTL)
    all_trials = pd.read_csv(os.path.join(exp_dir_raw, expID + '_all_trials.csv'))
    all_trials.insert(0,'time',trialOnsetTimesTL)
    all_trials.to_csv(os.path.join(exp_dir_processed, expID + '_all_trials.csv'), index=False)
    print('Done without errors')

    # for debugging:
def main():
    userID = 'adamranson'
    expID = '2024-07-17_01_ESMT170'
    run_preprocess_bv(userID, expID)

if __name__ == "__main__":
    main()