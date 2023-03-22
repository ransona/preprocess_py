import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy import interpolate
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib.pyplot as plt
import pickle
import organise_paths 
import preprocess_bv 
import preprocess_s2p
import preprocess_ephys

# expID
expID = '2023-02-28_11_ESMT116'
expID = '2023-03-01_01_ESMT107'
# user ID to use to place processed data
userID = 'adamranson'
skip_ca = False
animalID, remote_repository_root, \
    processed_root, exp_dir_processed, \
        exp_dir_raw = organise_paths.find_paths(userID, expID)

# make folder to store recordings after processing if needed
exp_dir_processed_recordings = os.path.join(processed_root, animalID, expID,'recordings')
os.makedirs(exp_dir_processed_recordings, exist_ok = True)

###########################################################
# Process bv data
###########################################################
# process bonvision related data, this includes relating bon vision time to TL time and wheel data
preprocess_bv.run_preprocess_bv(userID,expID)

###########################################################
# Process S2P data
###########################################################
# check suite2p folder exists to be processed
print('Starting S2P section...')
if os.path.exists(os.path.join(exp_dir_processed, 'suite2p')) and not skip_ca:
    preprocess_s2p.run_preprocess_s2p(userID, expID)

###########################################################
# Process ephys data
###########################################################
preprocess_ephys.run_preprocess_ephys(userID, expID)

###########################################################
# Process DLC data
###########################################################

####################################################
### cut up ephys, eye, and ca traces into trials ###
####################################################

