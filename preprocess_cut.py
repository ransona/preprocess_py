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

def sparse_cat_np(original_np,new_np):
    new_shape = new_np.shape
    if len(new_shape) == 1:
        # then it is going to be the wrong way around to be concatinated as a row
        new_np = new_np[np.newaxis,:]

    y_diff =  new_np.shape[1]-original_np.shape[1]

    if y_diff > 0:
        # new is bigger than original
        original_np = np.pad(original_np, ((0,0),(0,y_diff)), mode = 'constant', constant_values = np.nan)
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
        # load data s2p ca data
        pickle_in = open(os.path.join(exp_dir_processed_recordings,iS2P_file),"rb")
        ca_data = pickle.load(pickle_in)
        pickle_in.close
        # load oasis data
        oasis_filename = 's2p_oasis_' + iS2P_file[4:]
        try:
            pickle_in = open(os.path.join(exp_dir_processed_recordings,oasis_filename),"rb")
            oasis_data = pickle.load(pickle_in)
            pickle_in.close
            do_oasis = True
        except:
            print('OASIS data not found for ' + iS2P_file)
            do_oasis = False
    

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
        s2p_F_cut['F'] = np.full([roi_count, trial_count,max_snippet_len],np.nan,dtype=np.int16)
        s2p_Spikes_cut['Spikes'] = np.full([roi_count, trial_count,max_snippet_len],np.nan,dtype=np.float16)

        if do_oasis:
            oasis_dF_cut = {}
            oasis_spikes_cut = {}
            oasis_dF_cut['oasis_dF'] = np.full([roi_count, trial_count,max_snippet_len],np.nan,dtype=np.int16)
            oasis_spikes_cut['oasis_spikes'] = np.full([roi_count, trial_count,max_snippet_len],np.nan,dtype=np.uint16)

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
                # do the same for oasis data
                if do_oasis:
                    snippet_to_insert = oasis_data['oasis_dF'][iCell,first_sample:last_sample]
                    oasis_dF_cut['oasis_dF'][iCell,iTrial,0:len(snippet_to_insert)]=snippet_to_insert
                    snippet_to_insert = oasis_data['oasis_spikes'][iCell,first_sample:last_sample]
                    oasis_spikes_cut['oasis_spikes'][iCell,iTrial,0:len(snippet_to_insert)]=snippet_to_insert

        # debug:
        #plt.imshow(np.squeeze(ca_cut[0,:,:]),aspect='auto',cmap='gray', extent=[ca_cut_t[0],ca_cut_t[-1],-1,1],interpolation="nearest") 
        # save cut snippets for ch in pickle
        s2p_dF_cut['t'] = np.linspace(0,s2p_dF_cut['dF'].shape[2]/ca_framerate,s2p_dF_cut['dF'].shape[2]) - pre_time
        s2p_F_cut['t'] = s2p_dF_cut['t']
        s2p_Spikes_cut['t'] = s2p_dF_cut['t']
        with open(os.path.join(exp_dir_processed_cut,iS2P_file[0:7]+'_dF_cut.pickle'), 'wb') as f: pickle.dump(s2p_dF_cut, f)
        with open(os.path.join(exp_dir_processed_cut,iS2P_file[0:7]+'_F_cut.pickle'), 'wb') as f: pickle.dump(s2p_F_cut, f)
        with open(os.path.join(exp_dir_processed_cut,iS2P_file[0:7]+'_Spikes_cut.pickle'), 'wb') as f: pickle.dump(s2p_Spikes_cut, f)

        if do_oasis:
            oasis_dF_cut['t'] = s2p_dF_cut['t']
            oasis_spikes_cut['t'] = s2p_dF_cut['t']
            with open(os.path.join(exp_dir_processed_cut,iS2P_file[0:7]+'_oasis_dF_cut.pickle'), 'wb') as f: pickle.dump(oasis_dF_cut, f)
            with open(os.path.join(exp_dir_processed_cut,iS2P_file[0:7]+'_oasis_spikes_cut.pickle'), 'wb') as f: pickle.dump(oasis_spikes_cut, f)

    ### Cut ephys data ###
    ephys_combined = np.load(os.path.join(exp_dir_processed_recordings,'ephys.npy'))
    # loop through all trials collecting the ephys traces
    ephys_cut = {}
    for iTrial in range(all_trials.shape[0]):
        trial_onset_time = all_trials.loc[iTrial,'time']
        trial_end_time = trial_onset_time + all_trials.loc[iTrial,'duration']
        # collect samples from ephys
        first_sample = np.argmax(ephys_combined[0,:] >= (trial_onset_time-pre_time))
        last_sample = np.argmax(ephys_combined[0,:] >= (trial_end_time+pre_time))
        if iTrial == 0:
            ephys_cut['0'] = ephys_combined[np.newaxis,1,first_sample:last_sample]
            ephys_cut['1'] = ephys_combined[np.newaxis,2,first_sample:last_sample]
        else:
            ephys_cut['0'] = sparse_cat_np(ephys_cut['0'],ephys_combined[1,first_sample:last_sample])
            ephys_cut['1'] = sparse_cat_np(ephys_cut['1'],ephys_combined[2,first_sample:last_sample])
    
    ephys_cut['t'] = np.linspace(0,ephys_cut['0'].shape[1]/1000,ephys_cut['0'].shape[1])-pre_time
    # save in pickle
    with open(os.path.join(exp_dir_processed_cut,'ephys_cut.pickle'), 'wb') as f: pickle.dump(ephys_cut, f)

    ### Cut eye data ###
    # check DLC data exists
    if os.path.exists(os.path.join(exp_dir_processed_recordings,'dlcEyeLeft_resampled.pickle')) and os.path.exists(os.path.join(exp_dir_processed_recordings,'dlcEyeRight_resampled.pickle')):
        # load dlc data
        eyeDat_left = pickle.load(open(os.path.join(exp_dir_processed_recordings,'dlcEyeLeft_resampled.pickle'), "rb"))
        eyeDat_right = pickle.load(open(os.path.join(exp_dir_processed_recordings,'dlcEyeRight_resampled.pickle'), "rb"))
        # loop through all trials collecting the dlc traces
        eye_cut_left = {}
        eye_cut_right = {}
        for iTrial in range(all_trials.shape[0]):
            trial_onset_time = all_trials.loc[iTrial,'time']
            trial_end_time = trial_onset_time + all_trials.loc[iTrial,'duration']
            # collect samples from dlc
            first_sample = np.argmax(eyeDat_left['t'] >= (trial_onset_time-pre_time))
            last_sample = np.argmax(eyeDat_left['t'] >= (trial_end_time+post_time))
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
                # if the calibrated pupil data is in eyeDat then cut this too
                if eyeDat_left.get('x_d') is not None:
                    # we assume that if this field is here then all calibrated fields will be there
                    eye_cut_left['x_d'] = eyeDat_left['x_d'][np.newaxis,first_sample:last_sample]
                    eye_cut_left['y_d'] = eyeDat_left['y_d'][np.newaxis,first_sample:last_sample]
                    eye_cut_left['radius_d'] = eyeDat_left['radius_d'][np.newaxis,first_sample:last_sample]
                    eye_cut_left['velocity_d'] = eyeDat_left['velocity_d'][np.newaxis,first_sample:last_sample]
                    eye_cut_right['x_d'] = eyeDat_right['x_d'][np.newaxis,first_sample:last_sample]
                    eye_cut_right['y_d'] = eyeDat_right['y_d'][np.newaxis,first_sample:last_sample]
                    eye_cut_right['radius_d'] = eyeDat_right['radius_d'][np.newaxis,first_sample:last_sample]
                    eye_cut_right['velocity_d'] = eyeDat_right['velocity_d'][np.newaxis,first_sample:last_sample]          
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
                # if the calibrated pupil data is in eyeDat then cut this too
                if eyeDat_left.get('x_d') is not None:
                    # we assume that if this field is here then all calibrated fields will be there            
                    eye_cut_left['x_d'] = sparse_cat_np(eye_cut_left['x_d'],eyeDat_left['x_d'][np.newaxis,first_sample:last_sample])
                    eye_cut_left['y_d'] = sparse_cat_np(eye_cut_left['y_d'],eyeDat_left['y_d'][np.newaxis,first_sample:last_sample])
                    eye_cut_left['radius_d'] = sparse_cat_np(eye_cut_left['radius_d'],eyeDat_left['radius_d'][np.newaxis,first_sample:last_sample])
                    eye_cut_left['velocity_d'] = sparse_cat_np(eye_cut_left['velocity_d'],eyeDat_left['velocity_d'][np.newaxis,first_sample:last_sample])
                    eye_cut_right['x_d'] = sparse_cat_np(eye_cut_right['x_d'],eyeDat_right['x_d'][np.newaxis,first_sample:last_sample])
                    eye_cut_right['y_d'] = sparse_cat_np(eye_cut_right['y_d'],eyeDat_right['y_d'][np.newaxis,first_sample:last_sample])
                    eye_cut_right['radius_d'] = sparse_cat_np(eye_cut_right['radius_d'],eyeDat_right['radius_d'][np.newaxis,first_sample:last_sample])
                    eye_cut_right['velocity_d'] = sparse_cat_np(eye_cut_right['velocity_d'],eyeDat_right['velocity_d'][np.newaxis,first_sample:last_sample])
        # make time vector
        eye_sample_rate = 1/np.round(eyeDat_left['t'][1]-eyeDat_left['t'][0],4)
        eye_cut_left['t'] = np.linspace(0,eye_cut_left['x'].shape[1]/eye_sample_rate,eye_cut_left['x'].shape[1]) - pre_time
        eye_cut_right['t'] = np.linspace(0,eye_cut_right['x'].shape[1]/eye_sample_rate,eye_cut_right['x'].shape[1]) - pre_time
        # save in pickle
        with open(os.path.join(exp_dir_processed_cut,'eye_left_cut.pickle'), 'wb') as f: pickle.dump(eye_cut_left, f)        
        with open(os.path.join(exp_dir_processed_cut,'eye_right_cut.pickle'), 'wb') as f: pickle.dump(eye_cut_right, f)  
    else:
        print('DLC data not found')

    ### Cut wheel data ###
    # wheel['position']
    # wheel['speed']
    # wheel['t']
    wheel = pickle.load(open(os.path.join(exp_dir_processed_recordings,'wheel.pickle'), "rb"))
    # loop through all trials collecting the wheel traces
    wheel_cut = {}
    for iTrial in range(all_trials.shape[0]):
        trial_onset_time = all_trials.loc[iTrial,'time']
        trial_end_time = trial_onset_time + all_trials.loc[iTrial,'duration']
        # collect samples from wheel
        first_sample = np.argmax(wheel['t'] >= trial_onset_time - pre_time)
        last_sample = np.argmax(wheel['t'] >= trial_end_time + post_time)
        if iTrial == 0:
            wheel_cut['position'] = wheel['position'][np.newaxis,first_sample:last_sample]
            wheel_cut['speed'] = wheel['speed'][np.newaxis,first_sample:last_sample]
        else:
            wheel_cut['position'] = sparse_cat_np(wheel_cut['position'],wheel['position'][np.newaxis,first_sample:last_sample])
            wheel_cut['speed'] = sparse_cat_np(wheel_cut['speed'],wheel['speed'][np.newaxis,first_sample:last_sample])
    
    # make time vector
    wheel_sample_rate = 1/np.round(wheel['t'][1]-wheel['t'][0],4)
    wheel_cut['t'] = np.linspace(0,wheel_cut['position'].shape[1]/wheel_sample_rate,wheel_cut['position'].shape[1]) - pre_time
    # save in pickle
    with open(os.path.join(exp_dir_processed_cut,'wheel.pickle'), 'wb') as f: pickle.dump(wheel_cut, f)   

    print('done')

# for debugging:
def main():
    userID = 'pmateosaparicio'
    expID = '2025-07-07_05_ESPM154'
    pre_secs = 5
    post_secs = 5
    run_preprocess_cut(userID, expID, pre_secs, post_secs)

if __name__ == "__main__":
    main()



