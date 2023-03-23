# runs DLC for pupil detection
import os
import organise_paths
import sys 
import cv2
import deeplabcut

def crop_vids(userID, expID): 
    print('Cropping videos...')
    animalID, remote_repository_root, \
    processed_root, exp_dir_processed, \
        exp_dir_raw = organise_paths.find_paths(userID, expID)

    # Open the video
    eye_video_to_crop = os.path.join(exp_dir_raw,(expID + '_eye1.mp4'))
    cap = cv2.VideoCapture(eye_video_to_crop)

    # eye_video_to_crop=eye_video_to_crop.strip('.mp4')
    # Initialize frame counter
    cnt = 0

    # Some characteristics from the original video
    w_frame, h_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # Here you can define your croping values
    #x,y,h,w = 0,0,100,100
    x,y,h,w = 0,0,479,743 #left eye
    #x,y,h,w = 900,20,450,550 #right eye
    # output
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_filename = os.path.join(exp_dir_processed,(expID+'_eye1_left.avi'))
    out = cv2.VideoWriter(output_filename, fourcc, fps, (w, h))

    # Now we start
    while(cap.isOpened()):
        ret, frame = cap.read()

        cnt += 1 # Counting frames

        # Avoid problems when video finish
        if ret==True:
            # Croping the frame
            crop_frame = frame[y:y+h, x:x+w]


            # Percentage
            xx = cnt *100/frames
            print(int(xx),'%')

            # Saving from the desired frames
            # if 15 <= cnt <= 90:
            #    out.write(crop_frame)

            # I see the answer now. Here you save all the video
            out.write(crop_frame)

            # Just to see the video in real time          
            # cv2.imshow('frame',frame)
            # cv2.imshow('croped',crop_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    #eye_video_to_crop=select_file()
    cap = cv2.VideoCapture(eye_video_to_crop)
    #eye_video_to_crop=eye_video_to_crop.strip('.mp4')
    # Initialize frame counter
    cnt = 0

    # Some characteristics from the original video
    w_frame, h_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # Here you can define your croping values
    #x,y,h,w = 0,0,100,100
    #x,y,h,w = 0,0,479,743 #left eye
    x,y,h,w = 744,0,479,743 #right eye

    # output
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_filename = os.path.join(exp_dir_processed,(expID+'_eye1_right.avi'))
    out = cv2.VideoWriter(output_filename, fourcc, fps, (w, h))

    # Now we start
    while(cap.isOpened()):
        ret, frame = cap.read()
        cnt += 1 # Counting frames

        # Avoid problems when video finish
        if ret==True:
            # Croping the frame
            crop_frame = frame[y:y+h, x:x+w]
            crop_frame=cv2.flip(crop_frame,1)
            # Percentage
            xx = cnt *100/frames
            print(int(xx),'%')

            # Saving from the desired frames
            #if 15 <= cnt <= 90:
            #    out.write(crop_frame)

            # I see the answer now. Here you save all the video
            out.write(crop_frame)

            # Just to see the video in real time          
            # cv2.imshow('frame',frame)
            # cv2.imshow('croped',crop_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    out.release()


def run_preprocess_dlc(userID, expID): 
    animalID, remote_repository_root, \
        processed_root, exp_dir_processed, \
            exp_dir_raw = organise_paths.find_paths(userID, expID)

    # crop raw video into videos for each eye
    crop_vids(userID, expID)
    # crop_vids(userID, expID)
    config_path = '/data/common/dlc_models/eye/config.yaml'
    videos = os.path.join(exp_dir_processed,(expID+'_eye1_left.avi'))
    destfolder = exp_dir_processed
    deeplabcut.analyze_videos(config_path, videos, videotype='avi', shuffle=1, trainingsetindex=0, gputouse=None, save_as_csv=True, destfolder=destfolder, dynamic=(True, .5, 50))
    deeplabcut.create_labeled_video(config_path, videos, save_frames = True)

    videos= os.path.join(exp_dir_processed,(expID+'_eye1_right.avi'))
    destfolder = exp_dir_processed
    deeplabcut.analyze_videos(config_path, videos, videotype='avi', shuffle=1, trainingsetindex=0, gputouse=None, save_as_csv=True, destfolder=destfolder, dynamic=(True, .5, 50))
    deeplabcut.create_labeled_video(config_path, videos, save_frames = True)

# for debugging:
def main():
    # expID
    expID = '2023-02-28_11_ESMT116'
    expID = '2023-03-01_01_ESMT107'
    # user ID to use to place processed data
    userID = 'adamranson'
    run_preprocess_dlc(userID, expID)

if __name__ == "__main__":
    main()