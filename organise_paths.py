import os
import socket
import shutil
from pathlib import Path

def find_paths(userID, expID):
    computer_name = socket.gethostname()
    #print(computer_name)
    animalID = expID[14:]
    if computer_name == 'AdamDellXPS15' or computer_name == 'ar-lab-si2':
        # path to root of raw data (usually hosted on server
        remote_repository_root = os.path.normpath(os.path.join('P:/Pipeline/Repository'))
        # path to root of processed data
        processed_root = os.path.normpath(os.path.join('P:/Pipeline/Repository_Processed/',userID,'data/Repository'))
        # complete path to processed experiment data
        exp_dir_processed = os.path.normpath(os.path.join(processed_root, animalID, expID)  )
        # complete path to processed experiment data recordings
        exp_dir_processed_recordings = os.path.normpath(os.path.join(processed_root, animalID, expID,'recordings'))
        # complete path to raw experiment data (usually hosted on server)
        exp_dir_raw = os.path.normpath(os.path.join(remote_repository_root, animalID, expID))
    elif computer_name == 'dream':
        # assume server
        # path to root of raw data (usually hosted on server)
        remote_repository_root = os.path.join('/data/Remote_Repository')
        # path to root of processed data
        processed_root = os.path.join('/home/',userID,'data/Repository')
        # complete path to processed experiment data
        exp_dir_processed = os.path.join(processed_root, animalID, expID)  
        # complete path to processed experiment data recordings
        exp_dir_processed_recordings = os.path.join(processed_root, animalID, expID,'recordings')
        # complete path to raw experiment data (usually hosted on server)
        exp_dir_raw = os.path.join(remote_repository_root, animalID, expID)        


    return animalID, remote_repository_root, processed_root, exp_dir_processed,exp_dir_raw

def queue_path():
    computer_name = socket.gethostname()
    #print(computer_name)
    if computer_name == 'AdamDellXPS15' or computer_name == 'ar-lab-si2':
        # adam's laptop
        return 'P://Pipeline//queues//step1'
    elif computer_name == 'dream':
        # assume server
        return '/data/common/queues/step1'
    
def remote_queue_path():
    computer_name = socket.gethostname()
    #print(computer_name)
    if computer_name == 'AdamDellXPS15' or computer_name == 'ar-lab-si2':
        # adam's laptop
        return '/data/common/local_pipelines/' + computer_name + '/queues/step1'
    elif computer_name == 'dream':
        # assume server
        return '/data/common/queues/step1'   
        
def log_path(log_filename):
    computer_name = socket.gethostname()
    #print(computer_name)
    if computer_name == 'AdamDellXPS15' or computer_name == 'ar-lab-si2':
        # adam's laptop
        return os.path.join('P://Pipeline//queues//step1//logs', log_filename)
    elif computer_name == 'dream':
        # assume server
        return os.path.join('/data/common/queues/step1/logs/', log_filename)

def s2p_config_root():
    computer_name = socket.gethostname()
    #print(computer_name)
    if computer_name == 'AdamDellXPS15' or computer_name == 'ar-lab-si2':
        # adam's laptop
        return 'P://Pipeline//s2p_configs','/data/common/configs/s2p_configs'
    elif computer_name == 'dream':
        # assume server
        return '/data/common/configs/s2p_configs','/data/common/configs/s2p_configs'

def s2p_config_path(user,config_name):
    computer_name = socket.gethostname()
    #print(computer_name)
    if computer_name == 'AdamDellXPS15' or computer_name == 'ar-lab-si2':
        # adam's laptop
        return os.path.normpath(os.path.join('P://Pipeline//s2p_configs',user,config_name))
    elif computer_name == 'dream':
        # assume server
        return os.path.normpath(os.path.join('/data/common/configs/s2p_configs',user,config_name))

def s2p_launcher_command(run_as_user,userID, expID, suite2p_env, tif_path, s2p_output_path, config_path):

    computer_name = socket.gethostname()
    #print(computer_name)
    if computer_name == 'AdamDellXPS15':
        # adam's laptop
        s2p_launcher = os.path.normpath('P://code//repos//preprocess_py//s2p_launcher_meso.py')
        cmd = [
            r'C:\Users\ranso\anaconda3\envs\suite2p\python.exe',
            '-u',
            s2p_launcher,
            userID, expID, tif_path, s2p_output_path, config_path
        ]        
        return cmd
    elif computer_name == 'ar-lab-si2':
        # SI2 computer
        s2p_launcher = os.path.normpath('P://code//repos//preprocess_py//s2p_launcher_meso.py')
        cmd = [
            r'C:\Users\ScanImage\miniconda3\envs\suite2p\python.exe',
            '-u',
            s2p_launcher,
            userID, expID, tif_path, s2p_output_path, config_path
        ]        
        return cmd
    elif computer_name == 'dream':
        # assume server
        if run_as_user:
            cmd = ['sudo', '-u', userID, '/opt/scripts/conda-run.sh',suite2p_env,'python',s2p_launcher,userID,expID,tif_path,config_path]
        else:
            cmd = ['/opt/scripts/conda-run.sh',suite2p_env,'python',s2p_launcher,userID,expID,tif_path,s2p_output_path,config_path]
            
        return cmd
    
def get_local_s2p_path(expID):
    # used for running pipeline locally on DAQ computers
    computer_name = socket.gethostname()
    animalID = expID[14:]
    #print(computer_name)
    if computer_name == 'AdamDellXPS15':
        # adam's laptop
        return os.path.normpath(os.path.join('C:/Repository', animalID, expID))
    elif computer_name == 'ar-lab-si2':
        # scanimage computer in lab2
        return os.path.normpath(os.path.join('F:\Local_Repository', animalID, expID))    
    elif computer_name == 'dream':
        # assume server
        raise ValueError("Computer name not recognised. Please check the local_2p_path function.")
    
def get_nas_s2p_path(expID):
    # used for running pipeline locally on DAQ computers
    computer_name = socket.gethostname()
    animalID = expID[14:]
    #print(computer_name)
    if computer_name == 'AdamDellXPS15' or computer_name == 'ar-lab-si2':
        # adam's laptop
        return os.path.normpath(os.path.join('\\\\ar-lab-nas1\\DataServer\\Remote_Repository', animalID, expID))
    elif computer_name == 'dream':
        # assume server
        raise ValueError("Computer name not recognised. Please check the nas_s2p_path function.")
    
def remote_processed_data_root(jobID=None):
    # where to push data back to server so that it can then be moved to user folder
    computer_name = socket.gethostname()
    #print(computer_name)
    if computer_name == 'AdamDellXPS15' or computer_name == 'ar-lab-si2':
        # adam's laptop
        if jobID:
            return '/home/machine-pipeline-access/local_pipelines/' + computer_name + '/processed_data/' + jobID
        else:
            return '/home/machine-pipeline-access/local_pipelines/' + computer_name + '/processed_data/'
    elif computer_name == 'dream':
        ValueError("You are on dream. Please check the remote_processed_data_root function.")

def make_symbolic_links(expIDs, data_type):

    def find_tif_files(directory):
        directory = os.path.normpath(directory)
        if not os.path.isdir(directory):
            return [], []

        full_paths = []
        relative_paths = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith('.tif'):
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_path, directory)
                    full_paths.append(full_path)
                    relative_paths.append(relative_path)
        return full_paths, relative_paths

    def find_mp4_files(directory):
        if not os.path.isdir(directory):
            return [], []

        full_paths = []
        relative_paths = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith('.mp4'):
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_path, directory)
                    full_paths.append(full_path)
                    relative_paths.append(relative_path)
        return full_paths, relative_paths
     
    computer_name = socket.gethostname()
    #print(computer_name)
    if computer_name == 'AdamDellXPS15' or computer_name == 'ar-lab-si2':    
        # iterate over the expIDs checking if each exists, and if it does not make appropritate symbolic links
        if not ',' in expIDs:
            expIDs = [expIDs]
        else:
            expIDs = expIDs.split(',')

        
        for expID in expIDs:
            animalID = expID[14:]
            _, _, _, exp_dir_processed, exp_dir_raw = find_paths('null', expID)
            # remove exp_dir_raw if it exists even if it is empty
            if os.path.exists(exp_dir_raw):
                print(f"Removing existing raw data symbolic directory: {exp_dir_raw}")
                shutil.rmtree(exp_dir_raw)

            # check if the raw data exists
            if data_type == 's2p':
                # 1. check if there are tifs in the data folder locally
                local_2p_path  = get_local_s2p_path(expID)
                full_paths, relative_paths = find_tif_files(local_2p_path)
                if not full_paths:
                    # if not get path to nas
                    nas_2p_path  = get_nas_s2p_path(expID)
                    full_paths, relative_paths = find_tif_files(nas_2p_path)
                    if not full_paths:
                        # if not get path to server
                        # not yet implemented
                        raise ValueError("No tifs found in local, nas or server paths. Please check the paths.")
                    else:
                        print('Found tifs in nas path.')
                else:
                    print('Found tifs in local path.')
                
                # iterate over the tifs and make symbolic links to each
                for full_path, relative_path in zip(full_paths, relative_paths):
                        # make symbolic link to the tif in the processed folder
                        src = full_path
                        dest = os.path.join(exp_dir_raw, relative_path)
                        # make the directory if it does not exist
                        os.makedirs(os.path.dirname(dest), exist_ok=True)
                        if os.path.exists(dest):
                            os.unlink(dest)  # remove existing symbolic link or file
                        os.symlink(src, dest)
    elif computer_name == 'dream':
        print('Dream computer detected. No symbolic links made.')
                  
def get_ssh_settings():
    computer_name = socket.gethostname()
    #print(computer_name)
    if computer_name == 'AdamDellXPS15' or computer_name == 'ar-lab-si2':     
        host = '158.109.215.222'
        port = 10022
        username = 'machine-pipeline-access'
        key_path = '~/.ssh/id_ed25519_pipeline'
    elif computer_name == 'dream':
        ValueError('Dream computer detected. No symbolic links made.')
    else:
        raise ValueError("Computer name not recognised. Please check the get_ssh_settings function.")
    return host, port, username, key_path


def move_data_folder(src_path_str, job_id):
    src_path = Path(src_path_str)

    # Locate 'Repository_Processed' in the path
    parts = src_path.parts
    try:
        idx = parts.index("Repository_Processed")
    except ValueError:
        raise ValueError("'Repository_Processed' not found in source path")

    # Compute relative path after 'Repository_Processed'
    relative_path = Path(*parts[idx + 1:])

    # Compute new base path with job ID inserted
    dest_path = Path(*parts[:idx]) / "Repository_Processed" / "complete" / job_id / relative_path

    # Create required parent directories
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Move the folder if destination doesn't exist
    if not dest_path.exists():
        shutil.move(str(src_path), str(dest_path))
    else:
        raise FileExistsError(f"Destination already exists: {dest_path}")

    base_path = Path(*parts[:idx]) / "Repository_Processed" / "complete" / job_id
    return str(base_path)

# what to do when it runs as a script
if __name__ == "__main__":
    # example usage
    userID = 'user'
    expID = '2025-04-13_03_ESYB007'
    make_symbolic_links(expID,'s2p')