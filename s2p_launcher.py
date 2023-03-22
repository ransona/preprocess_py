# these scripts are to run commands that need to be run in specific conda environments
# they should be run from the command line
import sys
import suite2p
import organise_paths
import numpy as np
import subprocess
import os

# def run_s2p_launcher(userID, expID, suite2p_config): 
    #run_test_conda_run2(inputvar)
def main():
    print('Starting S2P Launcher...')
    userID = sys.argv[1]
    expID = sys.argv[2]
    tif_path = sys.argv[3]
    config_path = sys.argv[4]
    animalID, remote_repository_root, \
        processed_root, exp_dir_processed, \
            exp_dir_raw = organise_paths.find_paths(userID, expID)
    # load the saved config
    ops = np.load(config_path,allow_pickle=True)
    ops = ops.item()
    db = {
        'data_path': [tif_path],
        'save_path0': exp_dir_processed,
        'save_disk': exp_dir_processed, # where bin is moved after processing
        'fast_disk': os.path.join('/data/fast', animalID, expID),
        }
    
    output_ops = suite2p.run_s2p(ops=ops, db=db)  


if __name__ == "__main__":
    main()