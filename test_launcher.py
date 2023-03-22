# these scripts are to run commands that need to be run in specific conda environments
# they should be run from the command line
import sys
import suite2p
import organise_paths
import numpy as np
import subprocess
import os
import time

# def run_s2p_launcher(userID, expID, suite2p_config): 
    #run_test_conda_run2(inputvar)
def main():
    print('Starting TEST Launcher...')
    userID = sys.argv[1]
    expID = sys.argv[2]

    animalID, remote_repository_root, \
        processed_root, exp_dir_processed, \
            exp_dir_raw = organise_paths.find_paths(userID, expID)
    
    ops = suite2p.default_ops()
    print(userID)
    time.sleep(5)
    print(expID)


if __name__ == "__main__":
    main()