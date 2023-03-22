import sys
sys.path.append('~/code/preprocess_py')
import preprocess_step1
expIDs = ['2023-03-01_01_ESMT107','2023-03-01_02_ESMT107']
userID = 'adamranson'
suite2p_config = 'ch_1_depth_1.npy'

for expID in expIDs:
    print('Starting expID...')
    preprocess_step1.run_preprocess_step1(userID,expID,suite2p_config)