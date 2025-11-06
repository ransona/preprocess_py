# import os
# import subprocess
# import sys
# import pickle
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import organise_paths

# def run_command(cmd, step_name, log_file):
#     """Helper function to run a command and collect its output."""
#     output = []
#     print(f"Starting {step_name}", flush=True)
#     with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1) as proc:
#         for line in proc.stdout:
#             output.append(line)
#             print(line, end='', flush=True)  # Ensure immediate output
#         for line in proc.stderr:
#             output.append(f"Error: {line}")
#             print(f"Error: {line}", end='', flush=True)
#         proc.wait()
#     if proc.returncode != 0:
#         # Write collected logs before raising an exception
#         with open(log_file, 'a') as file:
#             file.write(f"--- {step_name} LOG START ---\n")
#             file.write("".join(output))
#             file.write(f"--- {step_name} LOG END ---\n")
#         raise Exception(f"An error occurred during the execution of {step_name}")
#     return "".join(output)

# def run_preprocess_step1(jobID, userID, expID, suite2p_config, runs2p, rundlc, runfitpupil): 
#     print('Starting job:', jobID)
#     print('--------------------------------------------------')

#     # Set up paths
#     if ',' not in expID:
#         # Single experiment ID
#         animalID, remote_repository_root, processed_root, exp_dir_processed, exp_dir_raw = organise_paths.find_paths(userID, expID)
#     else:
#         # Multiple experiment IDs
#         allExpIDs = expID.split(',')
#         tif_path = []
#         for expID_each in allExpIDs:
#             _, _, _, exp_dir_processed, exp_dir_raw = organise_paths.find_paths(userID, expID_each)
#             tif_path.append(exp_dir_raw)
#         exp_dir_raw = ','.join(tif_path)
#         _, _, _, exp_dir_processed, _ = organise_paths.find_paths(userID, allExpIDs[0])
    
#     # Ensure the output directory exists
#     os.makedirs(exp_dir_processed, exist_ok=True)

#     # Queue path and loading job config
#     queue_path = '/data/common/queues/step1'
#     queued_command = pickle.load(open(os.path.join(queue_path, jobID), "rb"))

#     # Check for suite2p environment
#     if 'suite2p_env' in queued_command['config']:
#         print('Suite2p env found')
#         print(f"Running as user {userID}")
#         suite2p_env = queued_command['config']['suite2p_env']
#         run_s2p_as_usr = True
#     else:
#         print('No suite2p env found - using default')
#         suite2p_env = 'suite2p'
#         run_s2p_as_usr = False

#     # Log file setup
#     log_file = os.path.join('/data/common/queues/step1/logs', f'{jobID[0:-1-6]}.txt')

#     # Prepare commands
#     commands = []
#     if runs2p:
#         # Suite2p command setup
#         s2p_launcher = os.path.join('/home', 'adamranson', 'code/preprocess_py/s2p_launcher.py')
#         config_path = os.path.join('/data/common/configs/s2p_configs', userID, suite2p_config)
#         if run_s2p_as_usr:
#             cmd = ['sudo', '-u', userID, '/opt/scripts/conda-run.sh', suite2p_env, 'python', s2p_launcher, userID, expID, exp_dir_raw, config_path]
#         else:
#             cmd = ['/opt/scripts/conda-run.sh', suite2p_env, 'python', s2p_launcher, userID, expID, exp_dir_raw, config_path]
#         commands.append(('suite2p', cmd))
    
#     if rundlc:
#         # DLC command setup
#         dlc_launcher = os.path.join('/home', 'adamranson', 'code/preprocess_py/dlc_launcher.py')
#         cmd = ['/opt/scripts/conda-run.sh', 'dlc-cuda', 'python', dlc_launcher, userID, expID]
#         commands.append(('DLC', cmd))
    
#     # Step 1: Run runs2p and rundlc in parallel and collect logs
#     task_logs = {}
#     with ThreadPoolExecutor() as executor:
#         future_to_command = {
#             executor.submit(run_command, cmd, name, log_file): name for name, cmd in commands
#         }
#         for future in as_completed(future_to_command):
#             step_name = future_to_command[future]
#             try:
#                 task_logs[step_name] = future.result()
#                 print(f"{step_name} completed successfully.")
#             except Exception as exc:
#                 task_logs[step_name] = f"{step_name} failed with exception: {exc}\n"
#                 with open(log_file, 'a') as file:
#                     for step, log in task_logs.items():
#                         file.write(f"--- {step} LOG START ---\n")
#                         file.write(log)
#                         file.write(f"--- {step} LOG END ---\n")
#                 raise exc
    
#     # Write all task logs to the log file
#     with open(log_file, 'a') as file:
#         for step, log in task_logs.items():
#             file.write(f"--- {step} LOG START ---\n")
#             file.write(log)
#             file.write(f"--- {step} LOG END ---\n")
    
#     # Step 2: Run runfitpupil after runs2p and rundlc are complete
#     if runfitpupil:
#         fit_pupil_launcher = os.path.join('/home', 'adamranson', 'code/preprocess_py/preprocess_pupil.py')
#         cmd = ['conda', 'run', '--no-capture-output', '--name', 'sci', 'python', fit_pupil_launcher, userID, expID]
#         print("Running Pupil Fit step...")
#         try:
#             result = run_command(cmd, "Pupil Fit", log_file)
#             with open(log_file, 'a') as file:
#                 file.write("--- Pupil Fit LOG START ---\n")
#                 file.write(result)
#                 file.write("--- Pupil Fit LOG END ---\n")
#             print("Pupil Fit completed successfully.")
#         except Exception as exc:
#             with open(log_file, 'a') as file:
#                 file.write("--- Pupil Fit LOG START ---\n")
#                 file.write(f"Pupil Fit failed with exception: {exc}\n")
#                 file.write("--- Pupil Fit LOG END ---\n")
#             raise exc

# # Sample usage
# def main():
#     jobID = '00_00_00_00_00_00_adamranson_2023-05-15_05_ESMT134.pickle'
#     expID = '2023-05-15_05_ESMT134,2023-05-15_06_ESMT134'
#     userID = 'adamranson'
#     suite2p_config = 'ch_1_2_depth_5_axon.npy'
#     runs2p = True
#     rundlc = True
#     runfitpupil = True
#     run_preprocess_step1(jobID, userID, expID, suite2p_config, runs2p, rundlc, runfitpupil)

# if __name__ == "__main__":
#     main()



import os
import organise_paths
import subprocess
import sys
import preprocess_pupil
import pickle
from datetime import datetime


def run_preprocess_step1(jobID,userID, expID, suite2p_config, runs2p, rundlc, runfitpupil): 
    print('Starting job: ' + jobID)
    print('--------------------------------------------------')
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
        # run as user method:
        # 1. load job config file
        # 2. determine if suite2p conda env option is set
        # 3. if set do:
        # env = 'suite2p'
        # cmd = ['sudo -u pmateosaparicio ', '/opt/scripts/conda-run.sh',env,'python',s2p_launcher,userID,expID,tif_path,config_path]
        queue_path = '/data/common/queues/step1'
        # debug:
        # queue_path = '/data/common/queues/step1/completed'
        # jobID = '2024_03_20_18_00_58_pmateosaparicio_2023-07-13_06_ESPM105'
        # end debug
        queued_command = pickle.load(open(os.path.join(queue_path,jobID), "rb"))
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
        
        if run_s2p_as_usr:
            cmd = ['sudo', '-u', userID, '/opt/scripts/conda-run.sh',suite2p_env,'python','-u',s2p_launcher,userID,expID,tif_path,config_path]
        else:
            cmd = ['/opt/scripts/conda-run.sh','suite2p','python','-u',s2p_launcher,userID,expID,tif_path,config_path]

        print('Starting S2P launcher...')
        now = datetime.now()
        print(now.strftime("%Y-%m-%d %H:%M:%S"))
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
    jobID = '00_00_00_00_00_00_adamranson_2023-05-15_05_ESMT134.pickle'
    expID = '2023-05-15_05_ESMT134,2023-05-15_06_ESMT134'
    userID = 'adamranson'
    suite2p_config = 'ch_1_2_depth_5_axon.npy'
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