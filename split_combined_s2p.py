import organise_paths
import os
import glob
import numpy as np
import shutil
import grp

def split_combined_suite2p():
    userID = 'adamranson'
    expID  = '2023-02-28_13_ESMT116'    # <--- put the first experiment of the sequence here
    animalID, remote_repository_root, \
        processed_root, exp_dir_processed, \
            exp_dir_raw = organise_paths.find_paths(userID, expID)
    suite2p_path = os.path.join(exp_dir_processed,'suite2p')
    suite2p_combined_path = os.path.join(exp_dir_processed,'suite2p_combined')
    if not os.path.exists(suite2p_combined_path):
        # Rename the suite2p folder in the first experiment's folder
        os.rename(suite2p_path, suite2p_combined_path)

    planes_list = glob.glob(os.path.join(suite2p_combined_path, '*plane*'))
    # determine all experiment IDs that have been combined
    combined_ops = np.load(os.path.join(exp_dir_processed,'suite2p_combined','plane0','ops.npy'),allow_pickle = True).item()
    iscell = np.load(os.path.join(exp_dir_processed,'suite2p_combined','plane0','iscell.npy'))

    expIDs = {}
    for iExp in range(len(combined_ops['data_path'])):
        expIDs[iExp] = os.path.basename(combined_ops['data_path'][iExp])

    all_animal_ids = []
    # check all experiments from the same animal
    for iExp in range(len(combined_ops['data_path'])):
        all_animal_ids.append(expIDs[iExp][14:])

    if len(set(all_animal_ids)) > 1:
        raise Exception('Combined multiple animals not permitted')

    for iPlane in range(len(planes_list)):
        print('Plane ' + str(iPlane))
        # load the combined data
        F = np.load(os.path.join(exp_dir_processed,'suite2p_combined','plane'+str(iPlane),'F.npy'))
        Fneu = np.load(os.path.join(exp_dir_processed,'suite2p_combined','plane'+str(iPlane),'Fneu.npy'))
        spks = np.load(os.path.join(exp_dir_processed,'suite2p_combined','plane'+str(iPlane),'spks.npy'))
        # iterate through experiments grabbing each's frames
        for iExp in range(len(expIDs)):
            expID = expIDs[iExp]
            frames_in_exp = combined_ops['frames_per_folder'][iExp]
            # calculate which frame in the combined data is the first from this experiment
            exp_start_frame = np.sum(combined_ops['frames_per_folder'][0:iExp]).astype(int)
            # select frames that come from this experiment
            F_exp = F[:,exp_start_frame:exp_start_frame+frames_in_exp-1]
            Fneu_exp = Fneu[:,exp_start_frame:exp_start_frame+frames_in_exp-1]
            spks_exp = spks[:,exp_start_frame:exp_start_frame+frames_in_exp-1]
            # save to experiment directory
            animalID2, remote_repository_root2, \
                processed_root2, exp_dir_processed2, \
                    exp_dir_raw2 = organise_paths.find_paths(userID, expID)
            # make output directory if it doesn't already exist
            os.makedirs(os.path.join(exp_dir_processed2,'suite2p','plane'+str(iPlane)), exist_ok = True)
            # save appropriate part of F etc
            print('Cropping and saving cell traces...')
            np.save(os.path.join(exp_dir_processed2,'suite2p','plane'+str(iPlane),'F.npy'),F_exp)
            np.save(os.path.join(exp_dir_processed2,'suite2p','plane'+str(iPlane),'Fneu.npy'),Fneu_exp)
            np.save(os.path.join(exp_dir_processed2,'suite2p','plane'+str(iPlane),'spks.npy'),spks_exp)
            # copy across iscell and ops files
            shutil.copy(os.path.join(exp_dir_processed,'suite2p_combined','plane'+str(iPlane),'iscell.npy'), \
                        os.path.join(exp_dir_processed2,'suite2p','plane'+str(iPlane),'iscell.npy'))
            shutil.copy(os.path.join(exp_dir_processed,'suite2p_combined','plane'+str(iPlane),'stat.npy'), \
                        os.path.join(exp_dir_processed2,'suite2p','plane'+str(iPlane),'stat.npy'))
            shutil.copy(os.path.join(exp_dir_processed,'suite2p_combined','plane'+str(iPlane),'ops.npy'), \
                        os.path.join(exp_dir_processed2,'suite2p','plane'+str(iPlane),'ops.npy'))
            # copy frames from registered video bin file to split folder
            print('Cropping and saving binary file (registered frames)...')
            path_to_source_bin = os.path.join(exp_dir_processed,'suite2p_combined','plane'+str(iPlane),'data.bin')
            path_to_dest_bin = os.path.join(exp_dir_processed2,'suite2p','plane'+str(iPlane),'data.bin')
            frameSize = combined_ops['meanImg'].shape
            frames_to_copy = range(exp_start_frame,exp_start_frame+frames_in_exp-1)
            split_s2p_vid(path_to_source_bin,path_to_dest_bin,frameSize,frames_to_copy,F.shape[1]);            
            # sort out permissions
        for iExp in range(len(expIDs)):
            expID = expIDs[iExp]
            animalID2, remote_repository_root2, \
                processed_root2, exp_dir_processed2, \
                    exp_dir_raw2 = organise_paths.find_paths(userID, expID)
            try:
                animalID, remote_repository_root, processed_root, exp_dir_processed, exp_dir_raw = organise_paths.find_paths(userID, expID)
                path = os.path.join(exp_dir_processed,'suite2p')
                group_id = grp.getgrnam('users').gr_gid
                mode = 0o770
                for root, dirs, files in os.walk(path):
                    for d in dirs:
                        dir_path = os.path.join(root, d)
                        os.chown(dir_path, -1, group_id)
                        os.chmod(dir_path, mode)
                    for f in files:
                        file_path = os.path.join(root, f)
                        os.chown(file_path, -1, group_id)
                        os.chmod(file_path, mode)
            except:
                print('Problem setting file permissions to user in step 1 batch')



def split_s2p_vid(path_to_source_bin, path_to_dest_bin, frameSize, frames_to_copy,total_frames):
    frames_to_copy = np.array(frames_to_copy, dtype=float)
    blockSize = 1000

    finfo = os.stat(path_to_source_bin)
    fsize = finfo.st_size
    frameCountCalculation = fsize / (frameSize[0] * frameSize[1] * 2)

    total_frames_to_write = len(frames_to_copy)

    frame_mean = []
    framesInSet = []

    with open(path_to_source_bin, 'rb') as fid, open(path_to_dest_bin, 'wb') as fid2:
        # jump forward in file to start of current experiment
        start_bytes = int(2 * frameSize[0] * frameSize[1] * frames_to_copy[0])
        fid.seek(start_bytes)

        for iStart in range(1, total_frames_to_write + 1, blockSize):
            lastFrame = iStart + blockSize - 1
            lastFrame = min(lastFrame, total_frames_to_write)
            framesToRead = lastFrame - iStart + 1
            print(f'Frame {iStart + frames_to_copy[0] - 1}-{lastFrame + frames_to_copy[0] - 1} of {frameCountCalculation}')

            # read block of frames
            print('Reading...')
            read_data = np.fromfile(fid, dtype=np.int16, count=frameSize[0]*frameSize[1]*framesToRead)

            # write to other file
            print('Writing...')
            read_data.tofile(fid2)

            # debug
            if iStart == 1:
                combined_data = read_data.reshape((frameSize[0], frameSize[1], framesToRead))

    combined_data = np.squeeze(np.mean(combined_data, axis=2))
    return combined_data
                                     
if __name__ == "__main__":
    split_combined_suite2p()