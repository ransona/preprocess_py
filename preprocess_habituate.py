
# function to take habituation recording and make a summary
# includes:
# - video motion energy vs time
# - binned pupil distribution
# - move experiment to habituation folder

import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import pickle
import shutil
import stat
import organise_paths


def apply_data_permissions_recursive(target_path):
    """Apply /data ownership and mode recursively to target_path."""
    data_stat = os.stat('/data')
    data_mode = stat.S_IMODE(data_stat.st_mode)
    data_uid = data_stat.st_uid
    data_gid = data_stat.st_gid

    for root, dirs, files in os.walk(target_path):
        os.chmod(root, data_mode)
        os.chown(root, data_uid, data_gid)
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            os.chmod(dir_path, data_mode)
            os.chown(dir_path, data_uid, data_gid)
        for file_name in files:
            file_path = os.path.join(root, file_name)
            os.chmod(file_path, data_mode)
            os.chown(file_path, data_uid, data_gid)


def preprocess_habituate_run(userID, expID):
    print('Starting preprocess_habituate_run...')
    animalID, remote_repository_root, \
    processed_root, exp_dir_processed, \
        exp_dir_raw = organise_paths.find_paths(userID, expID)
    exp_dir_processed_recordings = os.path.join(processed_root, animalID, expID,'recordings')


    # Move processed experiment to /data/common/habituation/<animalID>/
    habituation_root = os.path.join('/data', 'common', 'habituation')
    habituation_animal_dir = os.path.join(habituation_root, animalID)
    os.makedirs(habituation_animal_dir, exist_ok=True)

    exp_dir_name = os.path.basename(os.path.normpath(exp_dir_processed))
    exp_dir_processed_destination = os.path.join(habituation_animal_dir, exp_dir_name)

    if os.path.exists(exp_dir_processed):
        if os.path.exists(exp_dir_processed_destination):
            raise FileExistsError(
                f"Destination already exists: {exp_dir_processed_destination}"
            )
        shutil.move(exp_dir_processed, habituation_animal_dir)
        exp_dir_processed = exp_dir_processed_destination
    elif os.path.exists(exp_dir_processed_destination):
        exp_dir_processed = exp_dir_processed_destination
    else:
        raise FileNotFoundError(
            f"Processed experiment directory not found: {exp_dir_processed}"
        )

    # Match permissions and ownership to /data recursively
    apply_data_permissions_recursive(exp_dir_processed)



# for debugging:
def main():
        # debug mode
        print('Parameters received via debug mode')
        # # experiment lists
        allExpIDs = ['2025-11-28_02_ESRC026']
        userID = 'rubencorreia'   
        
        for expID in allExpIDs:
            preprocess_habituate_run(userID, expID)    

if __name__ == "__main__":
    main()
