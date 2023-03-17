import os
def find_paths(userID, expID):
    animalID = expID[14:]
    # path to root of raw data (usually hosted on gdrive but temporarily here)
    remote_repository_root = os.path.join('/data/Remote_Repository')
    # path to root of processed data
    processed_root = os.path.join('/home/',userID,'data/Repository')
    # complete path to processed experiment data
    exp_dir_processed = os.path.join(processed_root, animalID, expID)  
    # complete path to processed experiment data recordings
    exp_dir_processed_recordings = os.path.join(processed_root, animalID, expID,'recordings')
    # complete path to raw experiment data (usually hosted on gdrive)
    exp_dir_raw = os.path.join(remote_repository_root, animalID, expID)
    return animalID, remote_repository_root, processed_root, exp_dir_processed,exp_dir_raw