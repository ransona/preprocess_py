# these scripts are to run commands that need to be run in specific conda environments
# they should be run from the command line
import sys
import suite2p
import organise_paths
import numpy as np
import os
import socket
import shutil

if socket.gethostname() == 'AdamDellXPS15':
    print('Running on AdamDellXPS15 without thread limits')
elif socket.gethostname() == 'ar-lab-si2':  
    print('Running on ar-lab-si2 without thread limits')
else:
    print('Running on server without thread limits')
    # from conceivable import thread_limit

def s2p_launcher_run(userID,expID,tif_path,output_path,config_path):
    # determine if several experiments are being run together or not:

    # # remove any existing data
    # search_str = 'string_to_search'
    # for foldername in os.listdir(exp_dir_processed):
    #     if search_str in foldername and os.path.isdir(os.path.join(exp_dir_processed, foldername)):
    #         os.rmdir(os.path.join(exp_dir_processed, foldername))
    # split tif path: if there is only one path it still outputs this as a list
    # delete the output path and contents if it already exists using shutil.rmtree
    if os.path.exists(output_path):
        print('Deleting existing output path: ' + output_path)
        shutil.rmtree(output_path)

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
            'save_path0': output_path,
            #'save_disk': exp_dir_processed, # where bin is moved after processing
            #'fast_disk': os.path.join('/data/fast',userID, animalID, allExpIDs[0]),
            }
        ops['functional_chan']=1
        suite2p.run_s2p(ops=ops, db=db)  
        # run red ch
        # can be improved to avoid registering twice and making two copies of data!
        db = {
            'data_path': allTifPaths,
            'save_path0': os.path.join(output_path,'ch2')
            #'save_disk': exp_dir_processed, # where bin is moved after processing
            #'fast_disk': os.path.join('/data/fast',userID, animalID, allExpIDs[0]),
            }
        ops['functional_chan']=2
        suite2p.run_s2p(ops=ops, db=db)    
    else:
        # then we are running on 1 functional channel (this is a hack to encode this info)
        # run green ch
        db = {
            'data_path': allTifPaths,
            'save_path0': output_path,
            #'save_disk': exp_dir_processed, # where bin is moved after processing
            #'fast_disk': os.path.join('/data/fast',userID, animalID, allExpIDs[0]),
            }
        suite2p.run_s2p(ops=ops, db=db)  
            

# for debugging:
def main():
    print('S2P Launcher Run...')
    if len(sys.argv) > 1:
        try:
            'Run from command line'
            # has been run from sys command line after conda activate
            userID = sys.argv[1]
            print('userID = ' + userID)
            expID = sys.argv[2]
            print('ExpID = ' + expID)
            tif_path = sys.argv[3]
            print('tif_path = ' + tif_path)
            output_path = sys.argv[4]
            print('output_path = ' + output_path)
            config_path = sys.argv[5]
            print('config_path = ' + config_path)
            s2p_launcher_run(userID,expID,tif_path,output_path,config_path)
        except:
            # show the exception
            print('Error: ' + str(sys.exc_info()[0]))
            return
    else:
        # debug mode
        expID = '2025-04-13_03_ESYB007'
        userID = 'adamranson'        
        tif_path = 'C:\\Pipeline\\Repository\\ESYB007\\2025-04-13_03_ESYB007\\P2\\R001'
        output_path = 'C:\\Pipeline\\Repository_Processed\\adamranson\\data\\Repository\\ESYB007\\2025-04-13_03_ESYB007\\P2\\R001'
        config_path = os.path.join('C:\\Pipeline\\s2p_configs\\adamranson\\1c1d1024px.npy')
        s2p_launcher_run(userID,expID,tif_path,output_path,config_path)
    

if __name__ == "__main__":
    main()