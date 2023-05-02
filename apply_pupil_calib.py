
import time
start_time = time.time()
import pickle
import os
import organise_paths
import preprocess_cam
import preprocess_bv
import crop_videos
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shutil
from scipy.optimize import curve_fit
import cv2
from scipy.ndimage import median_filter


def apply_pupil_calib(userID, expIDs):
    # loop through each expID
    for expID in expIDs:
        print('################')
        print('Starting expID: ' + expID)
        animalID, remote_repository_root, processed_root, exp_dir_processed, exp_dir_raw = organise_paths.find_paths(userID, expID)
        exp_dir_processed_recordings = os.path.join(exp_dir_processed,'recordings')
        meta_data_path = os.path.join(processed_root,animalID,'meta')

        # load the pupil calibration file
        with open(os.path.join(meta_data_path,'eye_pix_angle_map.pickle'),'rb') as file: pupil_maps = pickle.load(file)

        # try to load the resampled eye data
        try:
            filenames = ['dlcEyeLeft_resampled.pickle','dlcEyeRight_resampled.pickle']
            for iFilename in range(len(filenames)):
                print('Starting ' + filenames[iFilename])
                with open(os.path.join(exp_dir_processed_recordings,filenames[iFilename]),'rb') as file: dlcEye_resampled = pickle.load(file)
                if iFilename == 0:
                    pupil_map = pupil_maps['left_map']
                else:
                    pupil_map = pupil_maps['right_map']

                # convert eye xy to degrees
                dlcEye_resampled['x_d'] = np.zeros(dlcEye_resampled['x'].shape)
                x = dlcEye_resampled['x']
                x[np.isnan(x)] = 0
                x[np.isnan(x)] = 0
                x = x.astype(np.int16)
                dlcEye_resampled['y_d'] = np.zeros(dlcEye_resampled['y'].shape)
                y = dlcEye_resampled['y']
                y[np.isnan(y)] = 0
                y[np.isnan(y)] = 0
                y = y.astype(np.int16)
                # use the lookup table to convert xy coordinates to elevation / azimuth
                dlcEye_resampled['x_d'] = pupil_map['azimuth'][y,x]
                dlcEye_resampled['x_d'][np.isnan(dlcEye_resampled['x'])] = np.nan
                dlcEye_resampled['y_d'] = pupil_map['elevation'][y,x]
                dlcEye_resampled['y_d'][np.isnan(dlcEye_resampled['y'])] = np.nan

                # plt.plot(dlcEye_resampled['x_d'])
                # plt.plot(dlcEye_resampled['y_d'])

                # for each pupil radius, extend the pupil x position by radius pixels to left and right 
                # and get the degrees of each of these extended positions. subtract one from the other 
                # to get the pupil diameter in degrees
                diameter = dlcEye_resampled['radius']
                diameter[np.isnan(diameter)] = 0
                diameter = diameter.astype(np.int16)
                pupil_left_deg = pupil_map['azimuth'][y,(x-(diameter/2)).astype(np.int16)]
                pupil_right_deg = pupil_map['azimuth'][y,(x+(diameter/2)).astype(np.int16)]
                pupil_diameter_deg = abs(pupil_left_deg - pupil_right_deg)
                pupil_diameter_deg[np.isnan(diameter)] = np.nan
                dlcEye_resampled['radius_d'] = pupil_diameter_deg/2

                # calculate pupil velocity
                xdiffs = np.diff(dlcEye_resampled['x_d'])
                ydiffs = np.diff(dlcEye_resampled['y_d'])
                eucla_diff = np.sqrt(xdiffs**2 + ydiffs**2)
                # calc distance moved in 10 samples = 1 second
                dlcEye_resampled['velocity_d'] = np.convolve(eucla_diff, np.ones(10), 'same')
                dlcEye_resampled['velocity_d'] = np.append(dlcEye_resampled['velocity_d'], dlcEye_resampled['velocity_d'][-1])
                dlcEye_resampled['velocity_d'] = np.array(dlcEye_resampled['velocity_d']) 

                print('Saving resampled pupil data with calibrated pupil position')
                with open(os.path.join(exp_dir_processed_recordings,filenames[iFilename]), 'wb') as f: pickle.dump(dlcEye_resampled, f) 
                with open(os.path.join(exp_dir_processed_recordings,filenames[iFilename]),'rb') as file: dlcEye_resampled2 = pickle.load(file)
                print('Done')

        except:
                print('Resampled eye data not found: no eye position calibration performed')

        


# for debugging:
def main():
    userID = 'adamranson'
    expID = ['2023-04-18_07_ESMT124']
    # expID = ['2023-02-24_02_ESMT116'] # lights off eye pos calib
    #expID = ['2023-04-18_01_ESMT124']
    #expID = ['2023-02-24_01_ESMT116']
    apply_pupil_calib(userID, expID)

if __name__ == "__main__":
    main()       