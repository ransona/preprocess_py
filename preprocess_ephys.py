import os
import numpy as np
from scipy.io import loadmat
import organise_paths

def run_preprocess_ephys(userID, expID):
    print('Starting run_preprocess_ephys...')
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

    ePhys1Idx = np.where(np.isin(tl_chNames, 'EPhys1'))
    ePhys2Idx = np.where(np.isin(tl_chNames, 'EPhys2'))
    ePhys1Data = np.squeeze(tl_daqData[:, ePhys1Idx])[np.newaxis,:]
    ePhys2Data = np.squeeze(tl_daqData[:, ePhys2Idx])[np.newaxis,:]
    ephys_combined = np.concatenate((tl_time,ePhys1Data,ePhys2Data),axis=0)
    np.save(os.path.join(exp_dir_processed_recordings,'ephys.npy'),ephys_combined)
    print('Done without errors')