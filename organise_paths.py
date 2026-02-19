import os
def find_paths(userID, expID):
    animalID = expID[14:]
    # Keep this root constant for all users.
    remote_repository_root = os.path.join('/data/Remote_Repository')
    if str(userID).lower() == 'habit':
        # Habituation data lives under a shared path per animal.
        processed_root = os.path.join('/data/common/habituation')
        exp_dir_processed = os.path.join(processed_root, animalID)
        exp_dir_raw = exp_dir_processed
    else:
        # path to root of processed data
        processed_root = os.path.join('/home/',userID,'data/Repository')
        # complete path to processed experiment data
        exp_dir_processed = os.path.join(processed_root, animalID, expID)
        # complete path to raw experiment data (usually hosted on gdrive)
        exp_dir_raw = os.path.join(remote_repository_root, animalID, expID)
    return animalID, remote_repository_root, processed_root, exp_dir_processed,exp_dir_raw
