import os
import subprocess
import sys
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import organise_paths  # Ensure this is available in your environment

def run_command(cmd, step_name):
    """Helper function to run a command and collect its output."""
    output = []
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1) as proc:
        for line in proc.stdout:
            output.append(line)
        for line in proc.stderr:
            output.append(f"Error: {line}")
        proc.wait()
    if proc.returncode != 0:
        raise Exception(f"An error occurred during the execution of {step_name}")
    return "".join(output)

def run_preprocess_step1(jobID, userID, expID, suite2p_config, runs2p, rundlc, runfitpupil): 
    print('Starting job:', jobID)
    print('--------------------------------------------------')

    # Set up paths
    if ',' not in expID:
        # Single experiment ID
        animalID, remote_repository_root, processed_root, exp_dir_processed, exp_dir_raw = organise_paths.find_paths(userID, expID)
    else:
        # Multiple experiment IDs
        allExpIDs = expID.split(',')
        tif_path = []
        for expID_each in allExpIDs:
            _, _, _, exp_dir_processed, exp_dir_raw = organise_paths.find_paths(userID, expID_each)
            tif_path.append(exp_dir_raw)
        exp_dir_raw = ','.join(tif_path)
        _, _, _, exp_dir_processed, _ = organise_paths.find_paths(userID, allExpIDs[0])
    
    # Ensure the output directory exists
    os.makedirs(exp_dir_processed, exist_ok=True)
    
    # Log file setup
    log_file = os.path.join('/data/common/queues/step1/logs', f'{jobID[0:-1-6]}.txt')

    # Prepare commands
    commands = []
    if runs2p:
        # Suite2p command setup
        s2p_launcher = os.path.join('/home', 'adamranson', 'code/preprocess_py/s2p_launcher.py')
        config_path = os.path.join('/data/common/configs/s2p_configs', userID, suite2p_config)
        cmd = ['/opt/scripts/conda-run.sh', 'suite2p', 'python', s2p_launcher, userID, expID, exp_dir_raw, config_path]
        commands.append(('suite2p', cmd))
    
    if rundlc:
        # DLC command setup
        dlc_launcher = os.path.join('/home', 'adamranson', 'code/preprocess_py/dlc_launcher.py')
        cmd = ['/opt/scripts/conda-run.sh', 'dlc-cuda', 'python', dlc_launcher, userID, expID]
        commands.append(('DLC', cmd))
    
    # Step 1: Run runs2p and rundlc in parallel and collect logs
    task_logs = {}
    with ThreadPoolExecutor() as executor:
        future_to_command = {
            executor.submit(run_command, cmd, name): name for name, cmd in commands
        }
        for future in as_completed(future_to_command):
            step_name = future_to_command[future]
            try:
                task_logs[step_name] = future.result()
                print(f"{step_name} completed successfully.")
            except Exception as exc:
                task_logs[step_name] = f"{step_name} failed with exception: {exc}\n"
                with open(log_file, 'a') as file:
                    for step, log in task_logs.items():
                        file.write(f"--- {step} LOG START ---\n")
                        file.write(log)
                        file.write(f"--- {step} LOG END ---\n")
                raise exc
    
    # Write all task logs to the log file
    with open(log_file, 'a') as file:
        for step, log in task_logs.items():
            file.write(f"--- {step} LOG START ---\n")
            file.write(log)
            file.write(f"--- {step} LOG END ---\n")
    
    # Step 2: Run runfitpupil after runs2p and rundlc are complete
    if runfitpupil:
        fit_pupil_launcher = os.path.join('/home', 'adamranson', 'code/preprocess_py/preprocess_pupil.py')
        cmd = ['conda', 'run', '--no-capture-output', '--name', 'sci', 'python', fit_pupil_launcher, userID, expID]
        print("Running Pupil Fit step...")
        try:
            result = run_command(cmd, "Pupil Fit")
            with open(log_file, 'a') as file:
                file.write("--- Pupil Fit LOG START ---\n")
                file.write(result)
                file.write("--- Pupil Fit LOG END ---\n")
            print("Pupil Fit completed successfully.")
        except Exception as exc:
            with open(log_file, 'a') as file:
                file.write("--- Pupil Fit LOG START ---\n")
                file.write(f"Pupil Fit failed with exception: {exc}\n")
                file.write("--- Pupil Fit LOG END ---\n")
            raise exc

# Sample usage
def main():
    jobID = '00_00_00_00_00_00_adamranson_2023-05-15_05_ESMT134.pickle'
    expID = '2023-05-15_05_ESMT134,2023-05-15_06_ESMT134'
    userID = 'adamranson'
    suite2p_config = 'ch_1_2_depth_5_axon.npy'
    runs2p = True
    rundlc = True
    runfitpupil = True
    run_preprocess_step1(jobID, userID, expID, suite2p_config, runs2p, rundlc, runfitpupil)

if __name__ == "__main__":
    main()
