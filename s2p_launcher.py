# these scripts are to run commands that need to be run in specific conda environments
# they should be run from the command line
import sys
import suite2p
import organise_paths
import numpy as np
import subprocess
import os

def s2p_launcher_run(userID,expID,tif_path,config_path):
    animalID, remote_repository_root, \
        processed_root, exp_dir_processed, \
            exp_dir_raw = organise_paths.find_paths(userID, expID)
    # # remove any existing data
    # search_str = 'string_to_search'
    # for foldername in os.listdir(exp_dir_processed):
    #     if search_str in foldername and os.path.isdir(os.path.join(exp_dir_processed, foldername)):
    #         os.rmdir(os.path.join(exp_dir_processed, foldername))
            
    # load the saved config
    ops = np.load(config_path,allow_pickle=True)
    ops = ops.item()
    db = {
        'data_path': [tif_path],
        'save_path0': exp_dir_processed,
        'save_disk': exp_dir_processed, # where bin is moved after processing
        'fast_disk': os.path.join('/data/fast',userID, animalID, expID),
        }
    
    output_ops = suite2p.run_s2p(ops=ops, db=db)  

# for debugging:
def main():
    print('Starting S2P Launcher...')
    try:
        # has been run from sys command line after conda activate
        userID = sys.argv[1]
        expID = sys.argv[2]
        tif_path = sys.argv[3]
        config_path = sys.argv[4]
    except:
        # debug mode
        expID = '2023-02-24_01_ESMT116'
        userID = 'adamranson'
        animalID, remote_repository_root, \
            processed_root, exp_dir_processed, \
                exp_dir_raw = organise_paths.find_paths(userID, expID)
        tif_path = exp_dir_raw
        config_path = os.path.join('/home',userID,'data/configs/s2p_configs','ch_1_depth_1.npy')

    s2p_launcher_run(userID,expID,tif_path,config_path)

if __name__ == "__main__":
    main()