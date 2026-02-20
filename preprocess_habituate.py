
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
import grp
import organise_paths


def apply_data_permissions_recursive(target_path):
    """Set group ownership to 'users' and grant group read/write recursively."""
    users_gid = grp.getgrnam("users").gr_gid

    def _apply_permissions(path):
        current_mode = stat.S_IMODE(os.stat(path).st_mode)
        group_rw = stat.S_IRGRP | stat.S_IWGRP
        if os.path.isdir(path):
            # Directories need execute bit for group traversal/access.
            os.chmod(path, current_mode | group_rw | stat.S_IXGRP)
        else:
            os.chmod(path, current_mode | group_rw)
        os.chown(path, -1, users_gid)

    _apply_permissions(target_path)
    for root, dirs, files in os.walk(target_path):
        for dir_name in dirs:
            _apply_permissions(os.path.join(root, dir_name))
        for file_name in files:
            _apply_permissions(os.path.join(root, file_name))


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
        allExpIDs = ['2026-01-19_01_ESRC026']
        userID = 'adamranson'   
        
        for expID in allExpIDs:
            preprocess_habituate_run(userID, expID)    

if __name__ == "__main__":
    main()
