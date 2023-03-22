import os, fnmatch
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy import interpolate
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy import signal
import sparse
import pickle
import organise_paths 

def sparse_cat_np(original_np,new_np):
    new_shape = new_np.shape
    if len(new_shape) == 1:
        # then it is going to be the wrong way around to be concatinated as a row
        new_np = new_np[np.newaxis,:]

    y_diff =  new_np.shape[1]-original_np.shape[1]

    if y_diff > 0:
        # new is bigger than original
        original_np = np.pad(original_np, (0,y_diff), 'constant', constant_values=np.nan)
    elif y_diff < 0:
        # new is smaller than original
        padding = np.empty([1,abs(y_diff)])
        padding[:] = np.nan
        new_np = np.concatenate([new_np, padding],axis = 1)
    
    # combine the two
    combined_np = np.concatenate([original_np,new_np],axis = 0)
    return combined_np

def run_preprocess_cut(userID, expID,pre_time,post_time):
    # work out path stuff
    animalID, remote_repository_root, \
        processed_root, exp_dir_processed, \
            exp_dir_raw = organise_paths.find_paths(userID, expID)
    exp_dir_processed_recordings = os.path.join(processed_root, animalID, expID,'recordings')
    exp_dir_processed_cut = os.path.join(exp_dir_processed,'cut')
    os.makedirs(exp_dir_processed_cut, exist_ok = True)
    # load trial data
    all_trials = pd.read_csv(os.path.join(exp_dir_processed, expID + '_all_trials.csv'))

    ### Cut Ca imaging data ###
    # load ca imaging traces
    # check if there are 1 or 2 channels
    all_s2p_files = fnmatch.filter(os.listdir(exp_dir_processed_recordings), 's2p_???.pickle')
    # cycle through each
    for iS2P_file in all_s2p_files:
        pickle_in = open(os.path.join(exp_dir_processed_recordings,iS2P_file),"rb")
        ca_data = pickle.load(pickle_in)
        pickle_in.close
        # preallocate an array with shape:
        # [roi,trials,time]
        ca_framerate = np.round(1/np.mean(np.diff(ca_data['t'])))
        roi_count = ca_data['dF'].shape[0]
        trial_count = all_trials.shape[0]
        max_snippet_len = int((pre_time+post_time+max(all_trials['duration']))*ca_framerate)
        ca_cut = np.full([roi_count, trial_count,max_snippet_len],np.nan,dtype=np.float16)
        ca_cut_t = np.arange(pre_time+post_time+max(all_trials['duration']))-pre_time
        for iTrial in range(all_trials.shape[0]):
            trial_onset_time = all_trials.loc[iTrial,'time']-pre_time
            trial_end_time = all_trials.loc[iTrial,'time'] + all_trials.loc[iTrial,'duration']+post_time
            # collect samples from ephys
            first_sample = np.argmax(ca_data['t'] >= trial_onset_time)
            last_sample = np.argmax(ca_data['t'] >= trial_end_time)
            if last_sample - first_sample > ca_cut.shape[2]:
                # make sure the snippet being cut will fit into the preallocated array and if not make it shorter (it should be about right already)
                last_sample = last_sample - ((last_sample - first_sample)-ca_cut.shape[2])
            for iCell in range(ca_data['dF'].shape[0]):
                # cut out snippets
                snippet_to_insert = ca_data['dF'][iCell,first_sample:last_sample]
                ca_cut[iCell,iTrial,0:len(snippet_to_insert)]=snippet_to_insert

        # debug:
        plt.imshow(np.squeeze(ca_cut),aspect='auto',cmap='gray', extent=[ca_cut_t[0],ca_cut_t[-1],-1,1])
        # save cut snippets for ch
        np.save(os.path.join(exp_dir_processed_cut,(iS2P_file[0:7]+'_cut.npy')), ca_cut)

    ### Cut ephys data ###
    ephys_combined = np.load(os.path.join(exp_dir_processed_recordings,'ephys.npy'))
    # loop through all trials collecting the ephys traces
    ephys_cut = {}
    for iTrial in range(all_trials.shape[0]):
        trial_onset_time = all_trials.loc[iTrial,'time']
        trial_end_time = trial_onset_time + all_trials.loc[iTrial,'duration']
        # collect samples from ephys
        first_sample = np.argmax(ephys_combined[0,:] >= trial_onset_time)
        last_sample = np.argmax(ephys_combined[0,:] >= trial_end_time)
        if iTrial == 0:
            ephys_cut[0] = ephys_combined[np.newaxis,1,first_sample:last_sample]
            ephys_cut[1] = ephys_combined[np.newaxis,2,first_sample:last_sample]
        else:
            ephys_cut[0] = sparse_cat_np(ephys_cut[0],ephys_combined[1,first_sample:last_sample])
            ephys_cut[1] = sparse_cat_np(ephys_cut[1],ephys_combined[2,first_sample:last_sample])
    
    # save in pickle
    with open(os.path.join(exp_dir_processed_cut,'ephys_cut.pickle'), 'wb') as f: pickle.dump(ephys_cut, f)

    ### Cut eye data ###
    # load dlc data
    pickle_in = open(os.path.join(exp_dir_processed_recordings,'dlcEyeLeft_resampled.pickle'),"rb")
    eyeDat_left = pickle.load(pickle_in)
    pickle_in.close()
    pickle_in = open(os.path.join(exp_dir_processed_recordings,'dlcEyeRight_resampled.pickle'),"rb")
    eyeDat_right = pickle.load(pickle_in)
    pickle_in.close()
    # loop through all trials collecting the dlc traces
    eye_cut_left = {}
    eye_cut_right = {}
    for iTrial in range(all_trials.shape[0]):
        trial_onset_time = all_trials.loc[iTrial,'time']
        trial_end_time = trial_onset_time + all_trials.loc[iTrial,'duration']
        # collect samples from dlc
        first_sample = np.argmax(eyeDat_left['t'] >= trial_onset_time)
        last_sample = np.argmax(eyeDat_left['t'] >= trial_end_time)
        if iTrial == 0:
            eye_cut_left['x'] = eyeDat_left['x'][np.newaxis,first_sample:last_sample]
            eye_cut_left['y'] = eyeDat_left['y'][np.newaxis,first_sample:last_sample]
            eye_cut_left['radius'] = eyeDat_left['radius'][np.newaxis,first_sample:last_sample]
            eye_cut_left['velocity'] = eyeDat_left['velocity'][np.newaxis,first_sample:last_sample]
            eye_cut_left['qc'] = eyeDat_left['qc'][np.newaxis,first_sample:last_sample]
            eye_cut_left['frame'] = eyeDat_left['frame'][np.newaxis,first_sample:last_sample]
            eye_cut_right['x'] = eyeDat_right['x'][np.newaxis,first_sample:last_sample]
            eye_cut_right['y'] = eyeDat_right['y'][np.newaxis,first_sample:last_sample]
            eye_cut_right['radius'] = eyeDat_right['radius'][np.newaxis,first_sample:last_sample]
            eye_cut_right['velocity'] = eyeDat_right['velocity'][np.newaxis,first_sample:last_sample]
            eye_cut_right['qc'] = eyeDat_right['qc'][np.newaxis,first_sample:last_sample]
            eye_cut_right['frame'] = eyeDat_right['frame'][np.newaxis,first_sample:last_sample]
        else:
            eye_cut_left['x'] = sparse_cat_np(eye_cut_left['x'],eyeDat_left['x'][np.newaxis,first_sample:last_sample])
            eye_cut_left['y'] = sparse_cat_np(eye_cut_left['y'],eyeDat_left['y'][np.newaxis,first_sample:last_sample])
            eye_cut_left['radius'] = sparse_cat_np(eye_cut_left['radius'],eyeDat_left['radius'][np.newaxis,first_sample:last_sample])
            eye_cut_left['velocity'] = sparse_cat_np(eye_cut_left['velocity'],eyeDat_left['velocity'][np.newaxis,first_sample:last_sample])
            eye_cut_left['qc'] = sparse_cat_np(eye_cut_left['qc'],eyeDat_left['qc'][np.newaxis,first_sample:last_sample])
            eye_cut_left['frame'] = sparse_cat_np(eye_cut_left['qc'],eyeDat_left['frame'][np.newaxis,first_sample:last_sample])
            eye_cut_right['x'] = sparse_cat_np(eye_cut_right['x'],eyeDat_right['x'][np.newaxis,first_sample:last_sample])
            eye_cut_right['y'] = sparse_cat_np(eye_cut_right['y'],eyeDat_right['y'][np.newaxis,first_sample:last_sample])
            eye_cut_right['radius'] = sparse_cat_np(eye_cut_right['radius'],eyeDat_right['radius'][np.newaxis,first_sample:last_sample])
            eye_cut_right['velocity'] = sparse_cat_np(eye_cut_right['velocity'],eyeDat_right['velocity'][np.newaxis,first_sample:last_sample])
            eye_cut_right['qc'] = sparse_cat_np(eye_cut_right['qc'],eyeDat_right['qc'][np.newaxis,first_sample:last_sample])
            eye_cut_right['frame'] = sparse_cat_np(eye_cut_right['qc'],eyeDat_right['frame'][np.newaxis,first_sample:last_sample])

    # save in pickle
    with open(os.path.join(exp_dir_processed_cut,'eye_left_cut.pickle'), 'wb') as f: pickle.dump(eye_cut_left, f)        
    with open(os.path.join(exp_dir_processed_cut,'eye_right_cut.pickle'), 'wb') as f: pickle.dump(eye_cut_right, f)  

    ### Cut video frame number data ###
    ephys_combined = np.load(os.path.join(exp_dir_processed_recordings,'ephys.npy'))
    # loop through all trials collecting the ephys traces
    ephys_cut = {}
    for iTrial in range(all_trials.shape[0]):
        trial_onset_time = all_trials.loc[iTrial,'time']
        trial_end_time = trial_onset_time + all_trials.loc[iTrial,'duration']
        # collect samples from ephys
        first_sample = np.argmax(ephys_combined[0,:] >= trial_onset_time)
        last_sample = np.argmax(ephys_combined[0,:] >= trial_end_time)
        if iTrial == 0:
            ephys_cut[0] = ephys_combined[np.newaxis,1,first_sample:last_sample]
            ephys_cut[1] = ephys_combined[np.newaxis,2,first_sample:last_sample]
        else:
            ephys_cut[0] = sparse_cat_np(ephys_cut[0],ephys_combined[1,first_sample:last_sample])
            ephys_cut[1] = sparse_cat_np(ephys_cut[1],ephys_combined[2,first_sample:last_sample])

    print('done')
# for debugging:
def main():
    # expID
    expID = '2023-02-28_11_ESMT116'
    expID = '2023-03-01_01_ESMT107'
    # user ID to use to place processed data
    userID = 'adamranson'
    run_preprocess_cut(userID, expID,2,2)

if __name__ == "__main__":
    main()



