import os, fnmatch
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy import interpolate
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy import signal
import pickle
import organise_paths 

def run_preprocess_trial_average(userID, expID,pre_time,post_time):
    # general purpose function which produces
    # 1. response pre and post amplitudes in a pre and post trial window for each neuron for each trial
    # 2. baseline subtracted response amplitudes for each trial
    # 3. max run speed in each trial, baseline and combined  period
    # pre_time is a list of 2 numbers specifying the range for the baseline period
    # post_time is a list of 2 numbers specifying the range for the stimulus period

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
        s2p_dF_cut = {}
        s2p_F_cut = {}
        s2p_Spikes_cut = {}
        s2p_dF_cut['dF'] = np.full([roi_count, trial_count,max_snippet_len],np.nan,dtype=np.float16)
        s2p_F_cut['F'] = np.full([roi_count, trial_count,max_snippet_len],np.nan,dtype=np.float16)
        s2p_Spikes_cut['Spikes'] = np.full([roi_count, trial_count,max_snippet_len],np.nan,dtype=np.float16)

        for iTrial in range(all_trials.shape[0]):
            trial_onset_time = all_trials.loc[iTrial,'time']
            trial_end_time = all_trials.loc[iTrial,'time'] + all_trials.loc[iTrial,'duration']
            # collect samples from ephys
            first_sample = np.argmax(ca_data['t'] >= (trial_onset_time-pre_time))
            last_sample = np.argmax(ca_data['t'] >= (trial_end_time+post_time))
            if last_sample - first_sample > s2p_dF_cut['dF'].shape[2]:
                # make sure the snippet being cut will fit into the preallocated array and if not make it shorter (it should be about right already)
                last_sample = last_sample - ((last_sample - first_sample)-s2p_dF_cut['dF'].shape[2])
            for iCell in range(ca_data['dF'].shape[0]):
                # cut out snippets
                snippet_to_insert = ca_data['dF'][iCell,first_sample:last_sample]
                s2p_dF_cut['dF'][iCell,iTrial,0:len(snippet_to_insert)]=snippet_to_insert
                snippet_to_insert = ca_data['F'][iCell,first_sample:last_sample]
                s2p_F_cut['F'][iCell,iTrial,0:len(snippet_to_insert)]=snippet_to_insert
                snippet_to_insert = ca_data['Spikes'][iCell,first_sample:last_sample]
                s2p_Spikes_cut['Spikes'][iCell,iTrial,0:len(snippet_to_insert)]=snippet_to_insert

        # debug:
        #plt.imshow(np.squeeze(ca_cut[0,:,:]),aspect='auto',cmap='gray', extent=[ca_cut_t[0],ca_cut_t[-1],-1,1],interpolation="nearest") 
        # save cut snippets for ch in pickle
        s2p_dF_cut['t'] = np.linspace(0,s2p_dF_cut['dF'].shape[2]/ca_framerate,s2p_dF_cut['dF'].shape[2]) - pre_time
        s2p_F_cut['t'] = s2p_dF_cut['t']
        s2p_Spikes_cut['t'] = s2p_dF_cut['t']

        with open(os.path.join(exp_dir_processed_cut,iS2P_file[0:7]+'_dF_cut.pickle'), 'wb') as f: pickle.dump(s2p_dF_cut, f)
        with open(os.path.join(exp_dir_processed_cut,iS2P_file[0:7]+'_F_cut.pickle'), 'wb') as f: pickle.dump(s2p_F_cut, f)
        with open(os.path.join(exp_dir_processed_cut,iS2P_file[0:7]+'_Spikes_cut.pickle'), 'wb') as f: pickle.dump(s2p_Spikes_cut, f)


# for debugging:
def main():
    userID = 'adamranson'
    expID = '2023-04-18_07_ESMT124'
    pre_secs = 5
    post_secs = 5
    run_preprocess_cut(userID, expID, pre_secs, post_secs)

if __name__ == "__main__":
    main()



