# from conceivable import thread_limit
import os
import organise_paths
import sys 
import cv2
import deeplabcut
import time
import shutil

def crop_vids(userID, expID): 
    print('Cropping videos...')
    animalID, remote_repository_root, \
    processed_root, exp_dir_processed, \
        exp_dir_raw = organise_paths.find_paths(userID, expID)
    # Decide which input file to use:
    # - Habituation setup (single eye): {expID}_habit.mp4
    # - Standard setup (two eyes):      {expID}_eye1.mp4
    habit_video = os.path.join(exp_dir_raw, (expID + '_habit.mp4'))
    two_eye_video = os.path.join(exp_dir_raw, (expID + '_eye1.mp4'))

    # Selection logic: prefer habit if it exists, otherwise fall back to two-eye.
    if os.path.exists(habit_video):
        eye_video_to_crop = habit_video
        is_habit = True
    else:
        eye_video_to_crop = two_eye_video
        is_habit = False

    # Open the source video once (previous code opened it twice).
    cap = cv2.VideoCapture(eye_video_to_crop)
    if not cap.isOpened():
        # Keep behavior explicit if the file can't be opened.
        print(f'Could not open video: {eye_video_to_crop}')
        return

    # Grab source properties once; outputs must match the source FPS.
    w_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # Hard-coded crop boxes (same values as before for two-eye).
    # NOTE: (x, y, h, w) ordering is preserved to avoid logic changes.
    left_crop = (0, 0, 479, 743)    # left eye in two-eye file
    right_crop = (744, 0, 479, 743) # right eye in two-eye file

    # Habituation file is a 640x480 single-eye video; crop = full frame.
    habit_crop = (0, 0, 480, 640)

    # Output labels (as requested).
    left_suffix = '_eye1_left'
    right_suffix = '_eye1_right'
    habit_suffix = '_eye1_right'

    # Prepare outputs once; write in a single read loop.
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # Build output writers based on selected input type.
    if is_habit:
        # Single right-eye output from full-frame crop.
        habit_output_filename = os.path.join(exp_dir_processed, (expID + habit_suffix + '.avi'))
        habit_out = cv2.VideoWriter(habit_output_filename, fourcc, fps, (habit_crop[3], habit_crop[2]))
        left_out = None
        right_out = None
    else:
        # Two-eye outputs (left + right) from the two-eye file.
        left_output_filename = os.path.join(exp_dir_processed, (expID + left_suffix + '.avi'))
        right_output_filename = os.path.join(exp_dir_processed, (expID + right_suffix + '.avi'))
        left_out = cv2.VideoWriter(left_output_filename, fourcc, fps, (left_crop[3], left_crop[2]))
        right_out = cv2.VideoWriter(right_output_filename, fourcc, fps, (right_crop[3], right_crop[2]))
        habit_out = None

    # Initialize frame counter and progress markers.
    cnt = 0
    progress_marks = [20, 40, 60, 80, 100]
    next_mark_index = 0

    # Read each frame once and write the relevant crops.
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # End of file or read error; match prior behavior by exiting loop.
            break

        cnt += 1

        if is_habit:
            # --- Habituation: single-eye full-frame crop ---
            x, y, h, w = habit_crop
            habit_frame = frame[y:y + h, x:x + w]
            habit_frame = cv2.flip(habit_frame, 1)
            habit_out.write(habit_frame)
        else:
            # --- Left eye crop (unchanged behavior) ---
            x, y, h, w = left_crop
            left_frame = frame[y:y + h, x:x + w]
            left_out.write(left_frame)

            # --- Right eye crop (unchanged behavior) ---
            x, y, h, w = right_crop
            right_frame = frame[y:y + h, x:x + w]
            right_frame = cv2.flip(right_frame, 1)
            right_out.write(right_frame)

        # Progress tracking: print at 20/40/60/80/100% once each.
        if frames:
            pct = cnt * 100 / frames
            if next_mark_index < len(progress_marks) and pct >= progress_marks[next_mark_index]:
                print(f'Cropping {progress_marks[next_mark_index]}% complete')
                next_mark_index += 1

        # Keep the quit key handling to preserve prior behavior.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources deterministically.
    cap.release()
    if left_out is not None:
        left_out.release()
    if right_out is not None:
        right_out.release()
    if habit_out is not None:
        habit_out.release()

def dlc_launcher_run(userID, expID):
    animalID, remote_repository_root, \
        processed_root, exp_dir_processed, \
            exp_dir_raw = organise_paths.find_paths(userID, expID)
    # make output directory if it doesn't already exist
    os.makedirs(exp_dir_processed, exist_ok = True)
    # removing all existing dlc data
    for filename in os.listdir(exp_dir_processed):
        if 'eye1_left' in filename:
            print('Deleting ' + filename)
            os.remove(os.path.join(exp_dir_processed, filename))
    # removing all existing dlc data
    for filename in os.listdir(exp_dir_processed):
        if 'eye1_right' in filename:
            print('Deleting ' + filename)
            os.remove(os.path.join(exp_dir_processed, filename))

    print('Starting cropping videos...')
    # crop raw video into videos for each eye
    crop_vids(userID, expID)

    config_path = '/data/common/dlc_models/all_setups-rubencorreia-2025-12-10/config.yaml'
    
    left_video = os.path.join(exp_dir_processed, (expID + '_eye1_left.avi'))
    destfolder = exp_dir_processed
    if os.path.exists(left_video):
        print('Starting left eye video...')
        deeplabcut.analyze_videos(config_path, left_video, shuffle=3, gputouse=0, save_as_csv=True, destfolder=destfolder)
    else:
        print('Skipping left eye video (file not found).')

    right_video = os.path.join(exp_dir_processed, (expID + '_eye1_right.avi'))
    destfolder = exp_dir_processed
    if os.path.exists(right_video):
        print('Starting right eye video...')
        deeplabcut.analyze_videos(config_path, right_video, shuffle=3, gputouse=0, save_as_csv=True, destfolder=destfolder)
    else:
        print('Skipping right eye video (file not found).')

    # If only right eye exists, duplicate outputs to create left-eye equivalents.
    if os.path.exists(right_video) and not os.path.exists(left_video):
        print('Only right eye present; duplicating DLC outputs for left eye...')

        # Create a fake left-eye video by copying the right-eye video.
        shutil.copy2(right_video, left_video)

        # Duplicate DLC outputs: any file with "all_setups" and "eye1_right"
        # gets a left-eye copy with "eye1_left" in the filename.
        for filename in os.listdir(exp_dir_processed):
            if 'all_setups' not in filename:
                continue
            if 'eye1_right' not in filename:
                continue
            src = os.path.join(exp_dir_processed, filename)
            dst_name = filename.replace('eye1_right', 'eye1_left')
            dst = os.path.join(exp_dir_processed, dst_name)
            shutil.copy2(src, dst)

    # If only left eye exists, duplicate outputs to create right-eye equivalents.
    if os.path.exists(left_video) and not os.path.exists(right_video):
        print('Only left eye present; duplicating DLC outputs for right eye...')

        # Create a fake right-eye video by copying the left-eye video.
        shutil.copy2(left_video, right_video)

        # Duplicate DLC outputs: any file with "all_setups" and "eye1_left"
        # gets a right-eye copy with "eye1_right" in the filename.
        for filename in os.listdir(exp_dir_processed):
            if 'all_setups' not in filename:
                continue
            if 'eye1_left' not in filename:
                continue
            src = os.path.join(exp_dir_processed, filename)
            dst_name = filename.replace('eye1_left', 'eye1_right')
            dst = os.path.join(exp_dir_processed, dst_name)
            shutil.copy2(src, dst)

# for debugging:
def main():
    print('Starting DLC Launcher...')
    try:
        # has been run from sys command line after conda activate
        userID = sys.argv[1]
        expID = sys.argv[2]
    except:
        # debug mode
        expID = '2026-01-20_02_ESRC027'
        userID = 'adamranson'
    start_time = time.time()
    dlc_launcher_run(userID, expID)
    print('Time to run: ' + str(time.time() - start_time) + ' secs')
if __name__ == "__main__":
    main()
