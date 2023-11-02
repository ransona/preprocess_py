import os
import organise_paths
import subprocess
import sys
import preprocess_pupil

def run_preprocess_step1(jobID,userID, expID, suite2p_config, runs2p, rundlc, runfitpupil): 
    if not ',' in expID:
        # then there's only one experiment ID
        animalID, remote_repository_root, \
            processed_root, exp_dir_processed, \
                exp_dir_raw = organise_paths.find_paths(userID, expID)
    else:
        # multiple expIDs being processing in suite2p together
        # make exp_dir_raw contain all data paths
        # make expID the first one
        allExpIDs = expID.split(',')
        # generate exp path for each expID
        tif_path = []
        for expID_each in allExpIDs:
            _, _, _, exp_dir_processed, exp_dir_raw = organise_paths.find_paths(userID, expID_each)
            tif_path.append(exp_dir_raw)
        exp_dir_raw = ','.join(tif_path)
        _, _, _, exp_dir_processed, _ = organise_paths.find_paths(userID, allExpIDs[0])


    # make the output directory if it doesn't already exist
    os.makedirs(exp_dir_processed, exist_ok = True)

    allOut = ''
    if runs2p:
        # run suite2p
        tif_path = exp_dir_raw
        config_path = os.path.join('/data/common/configs/s2p_configs',userID,suite2p_config)
        s2p_launcher = os.path.join('/home','adamranson', 'code/preprocess_py/s2p_launcher.py')
        #cmd = 'conda run --no-capture-output --name suite2p python '+ s2p_launcher +' "' + userID + '" "' + expID + '" "' + tif_path + '" "' + config_path + '"'
        # cmd = ['conda','run' , '--no-capture-output','--name','suite2p','python',s2p_launcher,userID,expID,tif_path,config_path]
        cmd = ['/opt/scripts/conda-run.sh','suite2p','python',s2p_launcher,userID,expID,tif_path,config_path]
        print('Starting S2P launcher...')
        #subprocess.run(cmd, shell=True)
        # with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:

        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1) as proc:
            # Read the output line by line and process it
            for line in proc.stdout:
                print(line)
                allOut = allOut + line
                sys.stdout.flush()
                with open('/data/common/queues/step1/logs/' + jobID[0:-1-6] + '.txt', 'a') as file:
                    file.write(line)

            # Read the error output line by line and process it
            error_output = ""
            for line in proc.stderr:
                print("Error: " + line)
                error_output += line
                sys.stdout.flush()
                with open('/data/common/queues/step1/logs/' + jobID[0:-1-6] + '.txt', 'a') as file:
                    file.write(line)

            proc.wait()
            if proc.returncode != 0:
                with open('/data/common/queues/step1/logs/' + jobID[0:-1-6] + '.txt', 'w') as file:
                    file.write(allOut)
                raise Exception("An error occurred during the execution of suite2p")


    if rundlc:
        # run DLC
        dlc_launcher = os.path.join('/home','adamranson', 'code/preprocess_py/dlc_launcher.py')
        print('Running DLC launcher...')
        # cmd = ['conda','run' , '--no-capture-output','--name','dlc-cuda','python',dlc_launcher,userID,expID]
        cmd = ['/opt/scripts/conda-run.sh','dlc-cuda','python',dlc_launcher,userID,expID]
        # Run the command
        #with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
            # Read the output line by line and process it
            for line in proc.stdout:
                print(line)
                allOut = allOut + line
                sys.stdout.flush()
                with open('/data/common/queues/step1/logs/' + jobID[0:-1-6] + '.txt', 'a') as file:
                    file.write(line)

            # # Read the error output line by line and process it
            # error_output = ""
            # for line in proc.stderr:
            #     print("Error: " + line)
            #     error_output += line
            #     with open('/data/common/queues/step1/logs/' + jobID[0:-1-6] + '.txt', 'a') as file:
            #         file.write(line)

            proc.wait()
            if proc.returncode != 0:
                # with open('/data/common/queues/step1/logs/' + jobID[0:-1-6] + '.txt', 'w') as file:
                #     file.write(allOut)
                raise Exception("An error occurred during the execution of dlc")

        # cmd = 'conda run --no-capture-output --name dlc-cuda python '+ dlc_launcher +' "' + userID + '" "' + expID + '"'
        # subprocess.run(cmd, shell=True)

    if runfitpupil:
        # run code to 
        # run code to take dlc output and fit circle to pupil etc
        fit_pupil_launcher = os.path.join('/home','adamranson', 'code/preprocess_py/preprocess_pupil.py')
        print('Running pupil fit launcher...')
        cmd = ['conda','run' , '--no-capture-output','--name','sci','python',fit_pupil_launcher,userID,expID]
        # Run the command
        #with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1) as proc:
            # Read the output line by line and process it
            for line in proc.stdout:
                print(line)
                allOut = allOut + line
                sys.stdout.flush()
                with open('/data/common/queues/step1/logs/' + jobID[0:-1-6] + '.txt', 'a') as file:
                    file.write(line)

            # Read the error output line by line and process it
            error_output = ""
            for line in proc.stderr:
                print("Error: " + line)
                error_output += line
                sys.stdout.flush()
                with open('/data/common/queues/step1/logs/' + jobID[0:-1-6] + '.txt', 'a') as file:
                    file.write(line)

            proc.wait()
            if proc.returncode != 0:
                with open('/data/common/queues/step1/logs/' + jobID[0:-1-6] + '.txt', 'w') as file:
                    file.write(allOut)
                raise Exception("An error occurred during the execution of pupil fit")

    # # save command line output
    # with open('/data/common/queues/step1/logs/' + jobID[0:-1-6] + '.txt', 'a') as file:
    #     file.write(allOut)

# for debugging:
def main():
    jobID = 'debug.pickle'
    expID = '2023-02-28_13_ESMT116,2023-02-28_14_ESMT116'
    userID = 'adamranson'
    suite2p_config = 'ch_1_depth_1.npy'
    runs2p          = True
    rundlc          = False
    runfitpupil     = False
    run_preprocess_step1(jobID,userID, expID,suite2p_config, runs2p, rundlc, runfitpupil)
    #run_preprocess_step1("debug","adamranson","2023-03-31_05_ESMT125","ch_1_depth_1.npy",False,True,True)
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