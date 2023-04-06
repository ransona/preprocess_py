# from conceivable import thread_limit
import os
import organise_paths
# import sys 
# import cv2
# import time

expID = '2023-04-04_04_ESMT125'
expID = '2023-04-04_04_ESMT124'
userID = 'adamranson'
animalID, remote_repository_root, \
    processed_root, exp_dir_processed, \
        exp_dir_raw = organise_paths.find_paths(userID, expID)

eye_video_to_crop = os.path.join(exp_dir_raw,(expID + '_eye1.mp4'))

print(os.path.isfile(eye_video_to_crop))

print(os.path.isfile('/data/Remote_Repository/ESMT124/2023-04-04_04_ESMT124/2023-04-04_04_ESMT124_eye1.mp4'))
