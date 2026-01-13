
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
import organise_paths


def preprocess_habituate_run(userID, expID):
    print('Starting preprocess_habituate_run...')
    animalID, remote_repository_root, \
    processed_root, exp_dir_processed, \
        exp_dir_raw = organise_paths.find_paths(userID, expID)
    exp_dir_processed_recordings = os.path.join(processed_root, animalID, expID,'recordings')

    if not os.path.exists(exp_dir_processed_recordings):
        os.mkdir(exp_dir_processed_recordings)

    # Create an empty marker file in the processed experiment directory
    os.makedirs(exp_dir_processed, exist_ok=True)
    empty_marker_path = os.path.join(exp_dir_processed, 'text.txt')
    with open(empty_marker_path, 'w'):
        pass



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
