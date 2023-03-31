import os
import organise_paths
import subprocess
import preprocess_pupil

def run_preprocess_step1(jobID,userID, expID, suite2p_config, runs2p, rundlc, runfitpupil): 
    animalID, remote_repository_root, \
        processed_root, exp_dir_processed, \
            exp_dir_raw = organise_paths.find_paths(userID, expID)
    allOut = ''
    if runs2p:
        # run suite2p
        tif_path = exp_dir_raw
        config_path = os.path.join('/home',os.getlogin(),'data/configs/s2p_configs',suite2p_config)
        s2p_launcher = os.path.join('/home',os.getlogin(), 'code/preprocess_py/s2p_launcher.py')
        #cmd = 'conda run --no-capture-output --name suite2p python '+ s2p_launcher +' "' + userID + '" "' + expID + '" "' + tif_path + '" "' + config_path + '"'
        cmd = ['conda','run' , '--no-capture-output','--name','suite2p','python',s2p_launcher,userID,expID,tif_path,config_path]
        print('Starting S2P launcher...')
        #subprocess.run(cmd, shell=True)
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
            # Read the output line by line and process it
            for line in proc.stdout:
                print(line)
                allOut = allOut + line

    if rundlc:
        # run DLC
        dlc_launcher = os.path.join('/home',os.getlogin(), 'code/preprocess_py/dlc_launcher.py')
        print('Running DLC launcher...')
        cmd = ['conda','run' , '--no-capture-output','--name','dlc-cuda','python',dlc_launcher,userID,expID]
        # Run the command
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
            # Read the output line by line and process it
            for line in proc.stdout:
                print(line)
                allOut = allOut + line
        # cmd = 'conda run --no-capture-output --name dlc-cuda python '+ dlc_launcher +' "' + userID + '" "' + expID + '"'
        # subprocess.run(cmd, shell=True)

    if runfitpupil:
        # run code to 
        # run code to take dlc output and fit circle to pupil etc
        fit_pupil_launcher = os.path.join('/home',os.getlogin(), 'code/preprocess_py/preprocess_pupil.py')
        print('Running pupil fit launcher...')
        cmd = ['conda','run' , '--no-capture-output','--name','sci','python',fit_pupil_launcher,userID,expID]
        # Run the command
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
            # Read the output line by line and process it
            for line in proc.stdout:
                print(line)
                allOut = allOut + line

    # save command line output
    with open('/data/common/queues/step1/logs/' + jobID[0:-1-6] + '.txt', 'w') as file:
        file.write(allOut)

# for debugging:
def main():
    jobID = 'debug.pickle'
    expID = '2023-02-24_01_ESMT116' # 7 tif
    expID = '2023-03-01_01_ESMT107' # 1 tif

    userID = 'adamranson'
    suite2p_config = 'ch_1_depth_1.npy'
    runs2p          = True
    rundlc          = True
    runfitpupil     = True
    #run_preprocess_step1(jobID,userID, expID,suite2p_config, runs2p, rundlc, runfitpupil)
    run_preprocess_step1("debug","pmateosaparicio","2022-02-08_03_ESPM040","ch_1_depth_1.npy",False,False,True)
    # cmd = ['python','/home/adamranson/code/preprocess_py/test_print.py']
    #pmateosaparicio_2022-02-08_03_ESPM040


    # # Run the command
    # with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
    #     # Read the output line by line and process it
    #     for line in proc.stdout:
    #         print(line)
    #         #print(line + ' saved')
    
if __name__ == "__main__":
    main()