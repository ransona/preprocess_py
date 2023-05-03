# cycles through experiment directory and subdirectories making a 
# list of all files and their total size, and saves this information
# in a csv file with a last line with a termination string

import hashlib
from pathlib import Path
import os
import sys

def hash_file(filepath):
    BUF_SIZE = 65536
    sha256 = hashlib.sha256()
    print('Hashing ' + str(filepath))
    with open(filepath, 'rb') as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()

def generate_file_data(local_repos_dir, output_stem,recursive_search):
    # local_repos_dir is the path to local repos of experiments
    output_file = 'file_check_' + output_stem + '.txt'

    local_repos_dir = Path(local_repos_dir)
    animal_list = []
    # create list of animal dirs
    for entry in local_repos_dir.iterdir():
        if entry.is_dir():
            animal_list.append(entry)
            
    if recursive_search:
        print('Running in recursive mode')
    # iterate through animal dirs, listing experiments and hashing each
    for iAnimal in range(len(animal_list)):
        animal_dir = Path(animal_list[iAnimal])
        exp_list = []
        animalID = os.path.basename(os.path.normpath(animal_dir))
        print('Starting animalID: ' + animalID)
        # create list of experiment dirs
        for entry in animal_dir.iterdir():
            if entry.is_dir():
                exp_list.append(entry)

        # iterate through experiments
        for exp_path in exp_list:
            expID = os.path.basename(os.path.normpath(exp_path))
            print('Starting expID: ' + expID)
            output_file_path = os.path.join(exp_path,output_file)
            # check if data in folder has already been hashed
            if not os.path.exists(output_file_path):
                # then hash all files
                root_path = Path(exp_path)
                file_data = []
                total_size = 0
                if recursive_search == True:
                    for filepath in root_path.glob('**/*'):
                        if filepath.is_file():
                            rel_path = filepath.relative_to(root_path).as_posix()
                            size = filepath.stat().st_size
                            file_hash = hash_file(filepath)
                            file_data.append((rel_path, size, file_hash))
                            total_size += size
                else:
                    for filepath in root_path.iterdir():
                        if filepath.is_file():
                            rel_path = filepath.relative_to(root_path).as_posix()
                            size = filepath.stat().st_size
                            file_hash = hash_file(filepath)
                            file_data.append((rel_path, size, file_hash))
                            total_size += size
                with open(output_file_path, 'w') as f:
                    f.write(f"Total size: {total_size}\n")
                    for rel_path, size, file_hash in file_data:
                        f.write(f"{rel_path}|{size}|{file_hash}\n")
            else:
                print('Skipping as hash file already found')
    print('All hashing complete')

# for debugging:
def main():
    try:
        # has been run from sys command line 
        root_path = sys.argv[1]
        output_file = sys.argv[2]
        recursive_search = bool(sys.argv[3])
    except:
        # debug        
        root_path = '/home/adamranson/temp/repos'
        output_file = 'scanimage'
        recursive_search = True
        
    generate_file_data(root_path,output_file,recursive_search)

if __name__ == "__main__":
    main()