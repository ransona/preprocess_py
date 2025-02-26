# step two takes checked suite2p and DLC output, and other experiment data 
# and gets it all ready for further analysis with:
# 1) finding when stims come on etc in tl time
# 2) finding ca imaging frame times in tl time
# 3) getting ephys data ready to use
# 4) getting DLC data ready to use
# 5) cutting traces from ca, dlc, and ephys into trials

import os
import organise_paths 
import preprocess_bv 
import preprocess_s2p
import preprocess_ephys
import preprocess_pupil_timestamp
import preprocess_cut
import preprocess_cam
import datetime

def run_preprocess_step2(userID, expID, pre_secs, post_secs, run_bonvision, run_s2p_timestamp, run_ephys, run_dlc_timestamp, run_cuttraces): 
    animalID, remote_repository_root, \
        processed_root, exp_dir_processed, \
            exp_dir_raw = organise_paths.find_paths(userID, expID)

    # make folder to store recordings after processing if needed
    exp_dir_processed_recordings = os.path.join(processed_root, animalID, expID,'recordings')
    if not os.path.exists(exp_dir_processed_recordings):
        os.makedirs(exp_dir_processed_recordings, exist_ok = True)

    if run_bonvision:
        ###########################################################
        # Process bv data
        ###########################################################
        # process bonvision related data, this includes relating bon vision time to TL time and wheel data
        print('Starting bonvision section...')
        # run version dependent on if we are using the new or old bonvision workflow
        exp_date = expID[0:10]
        exp_date = exp_date = expID[0:10]
        # test if experiment was performed after 2025-24-01
        if datetime.datetime.strptime(exp_date, '%Y-%m-%d') > datetime.datetime.strptime('2025-24-01', '%Y-%d-%m'):
            # run new bonvision workflow analysis
            preprocess_bv.run_preprocess_bv2(userID,expID) 
        else:
            # run old bonvision workflow analysis
            preprocess_bv.run_preprocess_bv(userID,expID)        
        

    if run_s2p_timestamp:
        ###########################################################
        # Process S2P data
        ###########################################################
        print('Starting S2P section...')
        preprocess_s2p.run_preprocess_s2p(userID, expID)

    if run_ephys:
        ###########################################################
        # Process ephys data
        ###########################################################
        print('Starting ephys section...')
        preprocess_ephys.run_preprocess_ephys(userID, expID)

    if run_dlc_timestamp:
        ###########################################################
        # Process DLC data (timestamping)
        ###########################################################
        print('Starting dlc timestamp section...')
        preprocess_cam.preprocess_cam_run(userID, expID)
        preprocess_pupil_timestamp.preprocess_pupil_timestamp_run(userID, expID)

    if run_cuttraces:
        ####################################################
        ### cut up ephys, eye, and ca traces into trials ###
        ####################################################
        print('Starting trail cutting section...')
        preprocess_cut.run_preprocess_cut(userID, expID, pre_secs, post_secs)
        