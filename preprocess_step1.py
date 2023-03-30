import os
import organise_paths
import subprocess
import preprocess_pupil

def run_preprocess_step1(userID, expID, suite2p_config, runs2p, rundlc, runfitpupil): 
    animalID, remote_repository_root, \
        processed_root, exp_dir_processed, \
            exp_dir_raw = organise_paths.find_paths(userID, expID)
    
    if runs2p:
        # run suite2p
        tif_path = exp_dir_raw
        config_path = os.path.join('/home',userID,'data/configs/s2p_configs',suite2p_config)
        s2p_launcher = os.path.join('/home',userID, 'code/preprocess_py/s2p_launcher.py')
        cmd = 'conda run --no-capture-output --name suite2p python '+ s2p_launcher +' "' + userID + '" "' + expID + '" "' + tif_path + '" "' + config_path + '"'
        print('Starting S2P launcher...')
        subprocess.run(cmd, shell=True)

    if rundlc:
        # run DLC
        dlc_launcher = os.path.join('/home',userID, 'code/preprocess_py/dlc_launcher.py')
        print('Running DLC launcher...')
        cmd = 'conda run --no-capture-output --name DEEPLABCUT python '+ dlc_launcher +' "' + userID + '" "' + expID + '"'
        subprocess.run(cmd, shell=True)

    if runfitpupil:
        # run code to 
        # run code to take dlc output and fit circle to pupil etc
        preprocess_pupil.preprocess_pupil_run(userID, expID)

# for debugging:
def main():
    expID = '2023-02-24_01_ESMT116'
    userID = 'adamranson'
    suite2p_config = 'ch_1_depth_1.npy'
    runs2p          = False
    rundlc          = True
    runfitpupil     = False
    run_preprocess_step1(userID, expID,suite2p_config, runs2p, rundlc, runfitpupil)

if __name__ == "__main__":
    main()