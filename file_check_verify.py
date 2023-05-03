# cycles through experiment directory and subdirectories making a 
# list of all files and their total size, and saves this information
# in a csv file with a last line with a termination string

import hashlib
from pathlib import Path
import os

def hash_file(filepath):
    BUF_SIZE = 65536
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()

def verify_file_data(file_data_stem, new_root_path):
    new_root_path = Path(new_root_path)
    total_size = 0
    stored_total_size = 0
    all_files_ok = True
    file_info_list = []
    file_data_path = 'file_check_' + file_data_stem + '.txt'
    file_data_path = os.path.join(new_root_path,file_data_path)

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
                else:
                    print(f"Size mismatch for {new_path}: expected {size} bytes, got {actual_size} bytes")
                    all_files_ok = False
            else:
                print(f"File not found: {new_path}")
                all_files_ok = False

    if all_files_ok and total_size == stored_total_size:
        for new_path, stored_hash in file_info_list:
            actual_hash = hash_file(new_path)
            if actual_hash != stored_hash:
                print(f"Hash mismatch for {new_path}: expected {stored_hash}, got {actual_hash}")
                all_files_ok = False
                break

        if all_files_ok:
            print(f"All files verified successfully: {total_size} bytes")
            output_file_path = Path(file_data_path)
            output_file_path_ok = output_file_path.with_stem(f"{output_file_path.stem}_ok")
            with open(output_file_path_ok, "w") as ok_file:
                ok_file.write("All files verified successfully")
        else:
            print(f"Verification failed: hash mismatch")
    else:
        print(f"Verification failed: expected {stored_total_size} bytes, got {total_size} bytes")


# for debugging:
def main():
    file_data_stem = 'scanimage'
    new_root_path = '/home/adamranson/temp/repos/A1/test_sync1' 
    verify_file_data(file_data_stem,new_root_path)

if __name__ == "__main__":
    main()