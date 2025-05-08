import os
import shutil
import stat
import time
import grp

def move_files_preserve_structure(source_root, dest_root):
    dest_mode = stat.S_IMODE(os.stat(dest_root).st_mode)

    for dirpath, _, filenames in os.walk(source_root):
        for filename in filenames:
            src_file = os.path.join(dirpath, filename)
            relative_path = os.path.relpath(src_file, source_root)
            dest_file = os.path.join(dest_root, relative_path)
            dest_dir = os.path.dirname(dest_file)

            if not os.path.exists(dest_file):
                os.makedirs(dest_dir, exist_ok=True)
                print(f'Moving {src_file} to {dest_file}')
                shutil.move(src_file, dest_file)
                print('ok')

def process_source_directories(base_source):
    for entry in os.listdir(base_source):
        source_subdir = os.path.join(base_source, entry)
        if not os.path.isdir(source_subdir):
            continue

        sync_complete_path = os.path.join(source_subdir, 'sync_complete')
        if not os.path.isfile(sync_complete_path):
            continue

        # Expecting exactly one directory inside this path
        inner_dirs = [d for d in os.listdir(source_subdir)
                      if os.path.isdir(os.path.join(source_subdir, d)) and d != 'sync_complete']
        if len(inner_dirs) != 1:
            continue  # Skip if not exactly one subdirectory

        user_dir = inner_dirs[0]
        source_user_path = os.path.join(source_subdir, user_dir)
        dest_user_path = os.path.join('/home', user_dir)

        if os.path.isdir(source_user_path):
            move_files_preserve_structure(source_user_path, dest_user_path)

if __name__ == "__main__":
    base_source_dir = "/home/machine-pipeline-access/data/local_pipelines/ar-lab-si2/processed_data"

    while True:
        process_source_directories(base_source_dir)
        time.sleep(600)
