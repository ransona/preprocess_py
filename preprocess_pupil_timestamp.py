# take dlc pipil output and fits circle to pupil etc

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import polygon
from circle_fit import taubinSVD as circle_fit
import time
import os
import pandas as pd
import pickle
import organise_paths
import apply_pupil_calib

def preprocess_pupil_timestamp_run(userID, expID):
    print('Starting preprocess_pupil_timestamp_run...')
    animalID, remote_repository_root, \
    processed_root, exp_dir_processed, \
        exp_dir_raw = organise_paths.find_paths(userID, expID)
    exp_dir_processed_recordings = os.path.join(processed_root, animalID, expID,'recordings')
    print('Starting ' + expID)
    dlc_filenames = [expID + '_eye1_leftDLC_resnet50_Trial_newMay19shuffle1_1030000.csv',
                expID + '_eye1_rightDLC_resnet50_Trial_newMay19shuffle1_1030000.csv']

    for iVid in range(0, len(dlc_filenames)):
        # load eyeDat which contains pupil position info derived from circles etc fit to dlc output
        if iVid == 0:
            eyeDat = pickle.load(open(os.path.join(exp_dir_processed_recordings,'dlcEyeLeft.pickle'), 'rb'))
        else:
            eyeDat = pickle.load(open(os.path.join(exp_dir_processed_recordings,'dlcEyeRight.pickle'), 'rb'))
        # store detected eye details with timeline timestamps
        # load video timestamps
        loggedFrameTimes = np.load(os.path.join(exp_dir_processed_recordings,'eye_frame_times.npy'))
        # resample to 10Hz constant rate
        newTimeVector = np.arange(round(loggedFrameTimes[0]), round(loggedFrameTimes[-1]), 0.1)
        frameVector = np.arange(0,len(eyeDat['x']))
        eyeDat2 = {}
        eyeDat2['t'] = newTimeVector
        eyeDat2['x'] = np.interp(newTimeVector, loggedFrameTimes, eyeDat['x'])
        eyeDat2['y'] = np.interp(newTimeVector, loggedFrameTimes, eyeDat['y'])
        eyeDat2['radius'] = np.interp(newTimeVector, loggedFrameTimes, eyeDat['radius'])
        eyeDat2['velocity'] = np.interp(newTimeVector, loggedFrameTimes, eyeDat['velocity'])
        eyeDat2['qc'] = np.interp(newTimeVector, loggedFrameTimes, eyeDat['qc'])
        eyeDat2['frame'] = np.round(np.interp(newTimeVector, loggedFrameTimes, frameVector))
        if iVid == 0:
            pickle_out = open(os.path.join(exp_dir_processed_recordings,'dlcEyeLeft_resampled.pickle'),"wb")
            pickle.dump(eyeDat2, pickle_out)
            pickle_out.close()
        else:
            pickle_out = open(os.path.join(exp_dir_processed_recordings,'dlcEyeRight_resampled.pickle'),"wb")
            pickle.dump(eyeDat2, pickle_out)
            pickle_out.close()           
        
    # check if there is a pix->degrees calibration file for the animal and if there is use it to convert
    # pupil xy position etc to degrees
    if os.path.exists(os.path.join(processed_root,animalID,'meta','eye_pix_angle_map.pickle')):
        # calibration exists so apply it
        print('Eye position calibration file found... applying calibration')
        apply_pupil_calib.apply_pupil_calib(userID,[expID]) 
    else:
        print('Warning: no eye position calibration file found')
    print()
    print('Done without errors')

# for debugging:
def main():
    userID = 'pmateosaparicio'
    expIDs = [
        '2025-07-04_04_ESPM154',    # stim
        '2025-07-07_05_ESPM154',    # stim
        '2025-07-02_03_ESPM135',    # stim
        '2025-07-08_04_ESPM152',    # stim
        '2025-07-11_02_ESPM154',    # stim
        '2025-07-04_06_ESPM154',    # sleep
        '2025-07-07_06_ESPM154',    # sleep
        '2025-07-02_05_ESPM135',    # sleep
        '2025-07-08_05_ESPM152',    # sleep
        '2025-07-11_03_ESPM154']    # sleep

    for expID in expIDs:
        preprocess_pupil_timestamp_run(userID, expID) 

# # experiment lists
# allExpIDs = ['2025-07-07_05_ESPM154']
# allExpIDs_sleep = [
#     '2025-07-04_06_ESPM154',
#     '2025-07-07_06_ESPM154',
#     '2025-07-11_03_ESPM154',    

#     for expID in allExpIDs:
#         preprocess_pupil_timestamp_run(userID, expID)   


if __name__ == "__main__":
    main()