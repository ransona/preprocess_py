
import os
import organise_paths
import subprocess
import sys
import preprocess_pupil
import pickle


def run_preprocess_step1_meso(jobID,userID, expID, suite2p_config, runs2p, rundlc, runfitpupil): 
    print('Starting job: ' + jobID)
    print('--------------------------------------------------')
    if jobID != '':
        queue_path = '/data/common/queues/step1'
        queued_command = pickle.load(open(os.path.join(queue_path,jobID), "rb"))
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
    else:
        print('No jobID provided - running in debug mode')
        exp_dir_raw = '/home/adamranson/data/tif_meso/local_repository/ESMT204/2025-03-05_02_ESMT204'
        exp_dir_processed = '/home/adamranson/data/tif_meso/processed_repository/ESMT204/2025-03-05_02_ESMT204'
        queued_command = {}
        queued_command['config'] = {}


    # make the output directory if it doesn't already exist
    os.makedirs(exp_dir_processed, exist_ok = True)

    # get the list format lists of suite2p configs
    suite2p_config = eval(suite2p_config)

    allOut = ''
    if runs2p:
        # 1) cycle through each path and roi folder running suite2p seperately
        # make list of all scan paths
        scanpath_names = []
        # cycle through checking if folders exist for each scan path from 0 to 9
        for i in range(10):
            # check if folder exists with name P + i
            path = os.path.join(exp_dir_raw, 'P' + str(i))
            if os.path.exists(path):
                # if it exists add to list
                scanpath_names.append(path)                    
        
        # within each scan path folder check what roi folders exist
        roi_folders = {}
        for i_scanpath in range(len(scanpath_names)):
            print('Starting scanpath ' + str(i_scanpath+1) + ' of ' + str(len(scanpath_names)))
            # list all roi folders
            roi_folders[i_scanpath] = []
            # store in roi_folders[i_scanpath] all roi folders within the scanpath folder
            roi_folders[i_scanpath] = sorted([f for f in os.listdir(scanpath_names[i_scanpath]) if os.path.isdir(os.path.join(scanpath_names[i_scanpath], f))])
            # iterate through all roi folders
            for i_roi in range(len(roi_folders[i_scanpath])):
                print('Starting roi ' + str(i_roi+1) + ' of ' + str(len(roi_folders[i_scanpath])))
                # form path the roi folder
                tif_path = os.path.join(exp_dir_raw,scanpath_names[i_scanpath],roi_folders[i_scanpath][i_roi])
                # form path to the s2p config file - First determine if there is a separate sweet 2P configuration for every ROI
                if len(suite2p_config[i_scanpath]) > 1:
                    # then there is a separate config for each roi
                    # check if there are enough configs for each roi
                    if len(suite2p_config[i_scanpath]) != len(roi_folders[i_scanpath]):
                        raise Exception("Not enough suite2p configs for each roi folder")
                    config_path = os.path.join('/data/common/configs/s2p_configs',userID,suite2p_config[i_scanpath][i_roi])
                else:
                    # then there is only one config for all rois
                    config_path = os.path.join('/data/common/configs/s2p_configs',userID,suite2p_config[i_scanpath][0])

                s2p_launcher = os.path.join('/home','adamranson', 'code/preprocess_py/s2p_launcher_meso.py')

                # check if a suite2p env has been set - if the suite2p from the launching users home will be used
                if 'suite2p_env' in queued_command['config']:
                    print('Suite2p env found')
                    print('Running as user ' + userID)
                    suite2p_env = queued_command['config']['suite2p_env']
                    run_s2p_as_usr = True
                else:
                    print('No suite2p env found - running in default adamranson:''suite2p env''')
                    suite2p_env = 'suite2p'
                    run_s2p_as_usr = False
                
                current_scanpath_name = os.path.basename(scanpath_names[i_scanpath])
                s2p_output_path = os.path.join(exp_dir_processed,current_scanpath_name,roi_folders[i_scanpath][i_roi])
                # make the output directory if it doesn't already exist
                os.makedirs(s2p_output_path, exist_ok = True)

                if run_s2p_as_usr:
                    cmd = ['sudo', '-u', userID, '/opt/scripts/conda-run.sh',suite2p_env,'python',s2p_launcher,userID,expID,tif_path,config_path]
                else:
                    cmd = ['/opt/scripts/conda-run.sh','suite2p','python',s2p_launcher,userID,expID,tif_path,s2p_output_path,config_path]

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
                
                x=0


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
    jobID = '2025_04_10_10_17_15_adamranson_2025-04-09_04_ESYB007.pickle'
    expID = '2025-04-09_04_ESYB007'
    userID = 'adamranson'
    suite2p_config = "[['ch_1_depth_1.npy'],['ch_1_depth_1.npy']]"
    runs2p          = True
    rundlc          = False
    runfitpupil     = False
    run_preprocess_step1_meso(jobID,userID, expID,suite2p_config, runs2p, rundlc, runfitpupil)
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