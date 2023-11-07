# these scripts are to run commands that need to be run in specific conda environments
# they should be run from the command line
from conceivable import thread_limit
import sys
import suite2p
import organise_paths
import numpy as np
import os

def s2p_launcher_run(userID,expID,tif_path,config_path):
    # determine if several experiments are being run together or not:

    # # remove any existing data
    # search_str = 'string_to_search'
    # for foldername in os.listdir(exp_dir_processed):
    #     if search_str in foldername and os.path.isdir(os.path.join(exp_dir_processed, foldername)):
    #         os.rmdir(os.path.join(exp_dir_processed, foldername))
    # split tif path: if there is only one path it still outputs this as a list
    allTifPaths = tif_path.split(',')
    print('tif_path = ' + tif_path)
    allExpIDs = expID.split(',')
    print('ExpID = ' + expID)
    animalID, remote_repository_root, processed_root, exp_dir_processed, exp_dir_raw = organise_paths.find_paths(userID, allExpIDs[0])       
    # load the saved config
    ops = np.load(config_path,allow_pickle=True)
    ops = ops.item()
    ops['save_mat'] = False
    if ops['functional_chan']==3:
        # then we are running on 2 functional channels (this is a hack to encode this info)
        db = {
            'data_path': allTifPaths,
            'save_path0': exp_dir_processed,
            #'save_disk': exp_dir_processed, # where bin is moved after processing
            #'fast_disk': os.path.join('/data/fast',userID, animalID, allExpIDs[0]),
            }
        suite2p.run_s2p(ops=ops, db=db)  
        # run red ch
        # can be improved to avoid registering twice and making two copies of data!
        db = {
            'data_path': allTifPaths,
            'save_path0': os.path.join(exp_dir_processed,'ch2')
            #'save_disk': exp_dir_processed, # where bin is moved after processing
            #'fast_disk': os.path.join('/data/fast',userID, animalID, allExpIDs[0]),
            }
        suite2p.run_s2p(ops=ops, db=db)    
    else:
        # then we are running on 1 functional channel (this is a hack to encode this info)
        # run green ch
        db = {
            'data_path': allTifPaths,
            'save_path0': exp_dir_processed,
            #'save_disk': exp_dir_processed, # where bin is moved after processing
            #'fast_disk': os.path.join('/data/fast',userID, animalID, allExpIDs[0]),
            }
        suite2p.run_s2p(ops=ops, db=db)  
            

# for debugging:
def main():
    print('Starting S2P Launcher...')
    try:
        # has been run from sys command line after conda activate
        userID = sys.argv[1]
        expID = sys.argv[2]
        tif_path = sys.argv[3]
        config_path = sys.argv[4]
    except:
        # debug mode
        expID = '2023-02-28_13_ESMT116'
        userID = 'adamranson'
        animalID, remote_repository_root, \
            processed_root, exp_dir_processed, \
                exp_dir_raw = organise_paths.find_paths(userID, expID)
        tif_path = '/data/Remote_Repository/ESMT116/2023-02-28_13_ESMT116,/data/Remote_Repository/ESMT116/2023-02-28_14_ESMT116'
        config_path = os.path.join('/home',userID,'data/configs/s2p_configs','ch_1_depth_1.npy')

    s2p_launcher_run(userID,expID,tif_path,config_path)

if __name__ == "__main__":
    main()