# script to copy already processed data from remote repos to personal folders

import os
import shutil
import organise_paths
import glob

def copy_directory(src_dir, dest_dir):
    # If the destination directory doesn't exist, create it
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for root, dirs, files in os.walk(src_dir):
        for file in files:
            src_path = os.path.join(root, file)
            dest_path = os.path.join(dest_dir, os.path.relpath(src_path, src_dir))

            # If the destination file exists and has a modification time
            # greater than or equal to the source file, skip this file
            if os.path.exists(dest_path) and os.path.getmtime(dest_path) >= os.path.getmtime(src_path):
                continue

            # Create the destination directory if it doesn't exist
            dest_dirname = os.path.dirname(dest_path)
            if not os.path.exists(dest_dirname):
                os.makedirs(dest_dirname)

            # Copy the file
            shutil.copy2(src_path, dest_path)

# for debugging:
def main():
    userID = 'adamranson'
    # expID
    exp_IDs_to_copy = {}
    exp_IDs_to_copy[0] = '2022-01-20_05_ESPM039'
    exp_IDs_to_copy[1] = '2022-01-21_03_ESPM039'
    exp_IDs_to_copy[2] = '2022-01-21_04_ESPM039'
    exp_IDs_to_copy[3] = '2022-01-28_02_ESPM039'
    exp_IDs_to_copy[4] = '2022-01-28_03_ESPM039'
    exp_IDs_to_copy[5] = '2022-02-07_03_ESPM039'
    exp_IDs_to_copy[6] = '2022-02-07_05_ESPM039'
    exp_IDs_to_copy[7] = '2022-03-17_02_ESPM039'
    exp_IDs_to_copy[8] = '2022-03-17_03_ESPM039'
    exp_IDs_to_copy[9] = '2022-03-17_04_ESPM039'
    exp_IDs_to_copy[10] = '2022-04-04_04_ESPM039'
    exp_IDs_to_copy[11] = '2022-04-04_05_ESPM039'
    exp_IDs_to_copy[12] = '2022-04-05_01_ESPM039'
    exp_IDs_to_copy[13] = '2022-04-05_02_ESPM039'
    exp_IDs_to_copy[14] = '2021-11-16_06_ESPM040'
    exp_IDs_to_copy[15] = '2021-11-16_08_ESPM040'
    exp_IDs_to_copy[16] = '2021-11-16_09_ESPM040'
    exp_IDs_to_copy[17] = '2022-02-08_03_ESPM040'
    exp_IDs_to_copy[18] = '2022-02-08_04_ESPM040'
    exp_IDs_to_copy[19] = '2022-06-15_02_ESPM062'
    exp_IDs_to_copy[20] = '2022-06-15_03_ESPM062'
    exp_IDs_to_copy[21] = '2022-06-10_02_ESPM065'
    exp_IDs_to_copy[22] = '2022-06-13_02_ESPM065'
    exp_IDs_to_copy[23] = '2022-06-13_03_ESPM065'
    exp_IDs_to_copy[24] = '2022-06-13_04_ESPM065'
    exp_IDs_to_copy[25] = '2022-09-23_02_ESPM094'
    exp_IDs_to_copy[26] = '2022-09-23_03_ESPM094'
    exp_IDs_to_copy[27] = '2022-09-23_04_ESPM094'
    exp_IDs_to_copy[28] = '2022-09-30_01_ESPM094'
    exp_IDs_to_copy[29] = '2022-09-30_02_ESPM094'

    for iExp in range(len(exp_IDs_to_copy)):
        expID = exp_IDs_to_copy[iExp]
        print(('Starting experiment ' + expID))
        animalID, remote_repository_root, \
            processed_root, exp_dir_processed, \
                exp_dir_raw = organise_paths.find_paths(userID, expID)
        if not os.path.exists(exp_dir_processed):
            os.makedirs(exp_dir_processed, exist_ok = True)
        # copy suite2p data
        try:
            if os.path.exists(os.path.join(exp_dir_raw,'suite2p')):
                print('Copying suite2p')
                copy_directory(os.path.join(exp_dir_raw,'suite2p'),os.path.join(exp_dir_processed,'suite2p'))
        except:
            print('Error copying suite2p')

        try:
            if os.path.exists(os.path.join(exp_dir_raw,'suite2p_combined')):
                print('Copying suite2p processed')
                copy_directory(os.path.join(exp_dir_raw,'suite2p_combined'),os.path.join(exp_dir_processed,'suite2p_combined'))
        except:
            print('Error copying suite2p')  

        # copy DLC data
        try:
            print('Copying DLC')
            dlc_filter = os.path.join(exp_dir_raw,'*resnet50*')
            for file in glob.glob(dlc_filter):
                shutil.copy(file, exp_dir_processed)
        except:
            print('Error copying dlc')  


if __name__ == "__main__":
    main()



