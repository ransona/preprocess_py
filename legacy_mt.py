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
    exp_list = '/home/adamranson/data/Configs/prepr_Exp_melina.txt'

    with open(exp_list, 'r') as file:
        # Read the first line from the file
        input_line = file.readline().strip()

    # Split the input line into separate values using the comma delimiter
    values = input_line.split(",")

    # Create an empty dictionary to store the key-value pairs
    exp_IDs_to_copy = {}

    # Iterate over the values and add them to the dictionary
    for i in range(len(values)):
        exp_IDs_to_copy[i] = values[i]

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



