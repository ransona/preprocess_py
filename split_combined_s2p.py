import organise_paths
import os
import glob
import numpy as np

def split_combined_suite2p():
    userID = 'pmateosaparicio'
    expID  = '2022-02-07_03_ESPM039'    
    animalID, remote_repository_root, \
        processed_root, exp_dir_processed, \
            exp_dir_raw = organise_paths.find_paths(userID, expID)
    suite2p_path = os.path.join(exp_dir_processed,'suite2p')
    suite2p_combined_path = os.path.join(exp_dir_processed,'suite2p_combined')
    if not os.path.exists(suite2p_combined_path):
        # Rename the suite2p folder in the first experiment's folder
        os.rename(suite2p_path, suite2p_combined_path)

    planes_list = glob.glob(os.path.join(suite2p_combined_path, '*plane*'))
    # determine all experiment IDs that have been combined
    combined_ops = np.load(os.path.join(exp_dir_processed,'suite2p_combined','plane0','ops.npy'),allow_pickle = True).item()
    expIDs = {}
    for iExp in range(len(combined_ops['data_path'])):
        expIDs[iExp] = os.path.basename(combined_ops['data_path'][iExp])

    all_animal_ids = []
    # check all experiments from the same animal
    for iExp in range(len(combined_ops['data_path'])):
        all_animal_ids.append(expIDs[iExp][14:])

    if len(set(all_animal_ids)) > 1:
        raise Exception('Combined multiple animals not permitted')

    

if __name__ == "__main__":
    split_combined_suite2p()