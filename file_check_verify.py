# cycles through experiment directory and subdirectories making a 
# list of all files and their total size, and saves this information
# in a csv file with a last line with a termination string

import hashlib
from pathlib import Path
import os

def hash_file(filepath):
    BUF_SIZE = 1073741824 # buffer 1GB at a time
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()

def verify_file_data(file_data_stem, new_root_path, output_root_path):
    new_root_path = Path(new_root_path)
    total_size = 0
    stored_total_size = 0
    all_files_ok = True
    file_info_list = []
    file_data_path = 'file_check_' + file_data_stem + '.txt'
    file_data_path = os.path.join(new_root_path,file_data_path)
    # check if the file integrity check file exists
    if not os.path.exists(file_data_path):
        print(file_data_stem + ' file integrity file not found (yet) for path ' + str(new_root_path))
        return(False,'File integrity check file not yet found')
    output_file_path_ok = 'file_check_' + file_data_stem + '.txt'
    output_file_path_ok = os.path.join(output_root_path,output_file_path_ok)    
    output_file_path_ok = Path(output_file_path_ok)
    output_file_path_ok = output_file_path_ok.with_stem(f"{output_file_path_ok.stem}_ok")
    # check if the files have already been verified 
    if not output_file_path_ok.exists():
        print('Checking ' + file_data_stem + ' file integrity for path ' + str(new_root_path))
        with open(file_data_path, 'r') as f:
            stored_total_size = int(f.readline().split(': ')[1])
            for line in f:
                rel_path, size, stored_hash = line.strip().split('|')
                size = int(size)
                new_path = new_root_path.joinpath(*Path(rel_path).parts)

                if new_path.exists():
                    actual_size = new_path.stat().st_size
                    if actual_size == size:
                        total_size += size
                        file_info_list.append((new_path, stored_hash))
                        print('.', end='', flush=True)
                    else:
                        print(f"Size mismatch for {new_path}: expected {size} bytes, got {actual_size} bytes")
                        all_files_ok = False
                else:
                    print(f"File not found: {new_path}")
                    all_files_ok = False

        
        if all_files_ok and total_size == stored_total_size:
            print('All file sizes correct - checking hash')
            for new_path, stored_hash in file_info_list:
                # could be improved to not hash files which have already passed the hash test
                # although if the size is correct most likely the hash is too
                actual_hash = hash_file(new_path)
                print('.', end='', flush=True)
                if actual_hash != stored_hash:
                    print(f"Hash mismatch for {new_path}: expected {stored_hash}, got {actual_hash}")
                    all_files_ok = False
                    break

            if all_files_ok:
                print(f"All files verified successfully: {total_size} bytes")
                # make exp folder if it doesn't already exist to output this ok file
                os.makedirs(Path(output_file_path_ok).parent, exist_ok=True)
                with open(output_file_path_ok, "w") as ok_file:
                    ok_file.write("All files verified successfully")
                return(True,'Hash correct')
            else:
                print(f"Verification failed: hash mismatch")
                # this indicates files are not verified
                return(False,'Hash mismatch')
        else:
            print(f"Verification failed: expected {stored_total_size} bytes, got {total_size} bytes")
            return(False,'Size mismatch')
    else:
        # files have already been verified
        # this indicates files are verifed but that this isn't new
        return(True,'Hash correct')


# for debugging:
def main():
    file_data_stem = 'test'
    new_root_path = '/home/adamranson/temp/test_sync0'
    verify_file_data(file_data_stem,new_root_path)

if __name__ == "__main__":
    main()