
import time
start_time = time.time()
import pickle
import os
import organise_paths
import preprocess_cam
import preprocess_bv
import crop_videos
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shutil
from scipy.optimize import curve_fit
import cv2
from scipy.ndimage import median_filter

print(f"Import time: {time.time() - start_time:.2f} seconds")

def polynomial_2D(x_y, a, b, c, d, e, f):
    x, y = x_y
    return a * x**2 + b * y**2 + c * x * y + d * x + e * y + f

def run_pupil_calib(userID, expIDs):
    # loop through each expID
    for expID in expIDs:
        print('################')
        print('Starting expID: ' + expID)
        animalID, remote_repository_root, processed_root, exp_dir_processed, exp_dir_raw = organise_paths.find_paths(userID, expID)
        exp_dir_processed_recordings = os.path.join(exp_dir_processed,'recordings')
        exp_dir_processed_cut = os.path.join(exp_dir_processed,'cut')
        video_path_left = os.path.join(exp_dir_processed,(expID+'_eye1_left.avi'))
        video_path_right = os.path.join(exp_dir_processed,(expID+'_eye1_right.avi'))

        # make meta data directory if it doesn't already exist
        meta_data_path = os.path.join(processed_root,animalID,'meta')
        os.makedirs(meta_data_path, exist_ok = True)

        # check prerequisite processing has been done and load required files
        # eyetracking frames have been timestamped?
        if os.path.exists(os.path.join(exp_dir_processed_recordings,'eye_frame_times.npy')):
            # load the eyetracking frames timestamps
            frame_times = np.load(os.path.join(exp_dir_processed_recordings,'eye_frame_times.npy'))
        else:
            # make folder for processed data
            os.makedirs(exp_dir_processed, exist_ok = True)
            os.makedirs(exp_dir_processed_recordings, exist_ok = True)
            # generate the eyetracking frame timestamps
            preprocess_cam.preprocess_cam_run(userID, expID)
            # load the eyetracking frames eye stamps
            frame_times = np.load(os.path.join(exp_dir_processed_recordings,'eye_frame_times.npy'))

        # stimulus information
        if os.path.exists(os.path.join(exp_dir_processed, expID + '_all_trials.csv')):
            # load stim info
            all_trials = pd.read_csv(os.path.join(exp_dir_processed, expID + '_all_trials.csv'))
        else:
            # generate / load stim info file
            preprocess_bv.run_preprocess_bv(userID,expID)
            all_trials = pd.read_csv(os.path.join(exp_dir_processed, expID + '_all_trials.csv'))

        # check if cropped videos are in the processed data directory and if not try to copy from the remote repos and if not then make them
        if (not os.path.isfile(video_path_left)) or (not os.path.isfile(video_path_right)):
            try:
                print('Copying eye videos - this is a one-off not required in future')
                shutil.copyfile(os.path.join(exp_dir_raw, (expID+'_eye1_left.avi')),video_path_left)
                shutil.copyfile(os.path.join(exp_dir_raw, (expID+'_eye1_right.avi')),video_path_right)
                print('Done!')
            except:
                # produce cropped videos
                crop_videos.crop_videos(userID, expID)

        cap = {0:cv2.VideoCapture(video_path_left),
            1:cv2.VideoCapture(video_path_right)}

        # loop through the two videos

        for iCap in [0,1]:
            # preload the whole video
            num_frames = int(cap[iCap].get(cv2.CAP_PROP_FRAME_COUNT))

            # Read the first frame
            cap[iCap].set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = cap[iCap].read()
            frame_size = np.array((frame.shape[0],frame.shape[1]))

            # Initialize an empty array to store the frames
            frames = np.zeros((num_frames,frame.shape[0],frame.shape[1]),np.uint8)

            # Loop through the video and append each frame to the list
            frame_num = 0

            # Record the start time
            start_time = time.time()

            # Decide part which will be set to zero
            xcrop = np.ceil(frame_size[1]*0.3).astype(int)

            print('Loading eye video...')
            while success:
                frame = np.sum(frame,axis=2)
                frames[frame_num,:,:] = frame
                success, frame = cap[iCap].read()
                frame_num = frame_num + 1

            vid_mean = np.mean(frames)
            frames[:,:,0:xcrop]=vid_mean.astype(int)

            # Print the elapsed time
            print(f"Video preload time: {time.time() - start_time:.2f} seconds")

            num_stims = len(all_trials['stim'].unique())
            trial_indices = all_trials.loc[(all_trials['stim'] == 1)].index

            all_stim_pos = np.full([num_stims,2],0)

            # stim times are:
            # 0 - 1 sec - black
            # 1 - 2 sec - black with white square

            start_time = time.time()
            # preallocate for average frames of each condition
            # frame_size = [200,200]
            mean_pretrial = np.full(np.append(num_stims,frame_size),0)
            mean_pre = np.full(np.append(num_stims,frame_size),0)
            mean_post = np.full(np.append(num_stims,frame_size),0)
            mean_diff = np.full(np.append(num_stims,frame_size),0)

            print('Cutting video...')

            for iStim in range(num_stims):

                trial_indices = all_trials.loc[(all_trials['stim'] == iStim+1)].index
                # store x an d y pos
                all_stim_pos[iStim,0] = all_trials.loc[trial_indices[0],'F2_x']
                all_stim_pos[iStim,1] = all_trials.loc[trial_indices[0],'F2_y']
                # iterate through trials of stim type
                # preallocate for space for trials within stim cond
                mean_pretrial_trials = np.full(np.append(len(trial_indices),frame_size),0)
                mean_pre_trials = np.full(np.append(len(trial_indices),frame_size),0)
                mean_post_trials = np.full(np.append(len(trial_indices),frame_size),0)

                for iTrial in range(len(trial_indices)):

                    # find pre trial frames
                    trial_onset_time = all_trials.loc[trial_indices[iTrial],'time']
                    pretrialframes = np.where((frame_times >= trial_onset_time - 0.5) & (frame_times <= trial_onset_time))
                    pretrialframes = pretrialframes[0]
                    preframes = np.where((frame_times >= trial_onset_time + 0.4) & (frame_times <= trial_onset_time + 0.8))
                    preframes = preframes[0]
                    postframes = np.where((frame_times >= trial_onset_time + 1.2) & (frame_times <= trial_onset_time + 1.8))
                    postframes = postframes[0]
                    # load the pretrial frames
                    pretrial_all_frames = np.full(np.append(len(pretrialframes),frame_size),0)
                    for iFrame in range(len(pretrialframes)):
                        #cap[iCap].set(cv2.CAP_PROP_POS_FRAMES, int(pretrialframes[iFrame]))
                        #ret, frame = cap[iCap].read()
                        #frame = np.sum(frame,axis=2)
                        # crop frame to eye
                        # frame = frame[200:400,350:550]
                        pretrial_all_frames[iFrame,:,:] = frames[pretrialframes[iFrame]]

                    # load the pre frames
                    pre_all_frames = np.full(np.append(len(preframes),frame_size),0)
                    for iFrame in range(len(preframes)):
                        #cap[iCap].set(cv2.CAP_PROP_POS_FRAMES, int(preframes[iFrame]))
                        #ret, frame = cap[iCap].read()
                        #frame = np.sum(frame,axis=2)
                        # crop frame to eye
                        # frame = frame[200:400,350:550]
                        pre_all_frames[iFrame,:,:] = frames[preframes[iFrame]]

                    # load the post frames
                    post_all_frames = np.full(np.append(len(postframes),frame_size),0)
                    for iFrame in range(len(postframes)):
                        #cap[iCap].set(cv2.CAP_PROP_POS_FRAMES, int(postframes[iFrame]))
                        #ret, frame = cap[iCap].read()
                        #frame = np.sum(frame,axis=2)
                        # crop frame to eye
                        # frame = frame[200:400,350:550]
                        post_all_frames[iFrame,:,:] = frames[postframes[iFrame]]

                    # store mean of the pre and post trial frames
                    mean_pretrial_trials[iTrial,:,:] = np.mean(pretrial_all_frames,axis = 0)
                    mean_pre_trials[iTrial,:,:] = np.mean(pre_all_frames,axis = 0)
                    mean_post_trials[iTrial,:,:] = np.mean(post_all_frames,axis = 0)
                    
                # store the mean of the pre and post trials
                mean_pretrial[iStim,:,:] = np.mean(mean_pretrial_trials,axis=0)
                mean_pre[iStim,:,:] = np.mean(mean_pre_trials,axis=0)
                mean_post[iStim,:,:] = np.mean(mean_post_trials,axis=0)
                mean_diff[iStim,:,:] = mean_post[iStim,:,:] - mean_pre[iStim,:,:]
                #plt.imshow(mean_diff[iStim,:,:])
                #plt.colorbar()
                #plt.show()
                # plot.
                # x=0

            print(f"Video cutting time: {time.time() - start_time:.2f} seconds")

            # display each stim cond
            sorted_indices = np.lexsort((all_stim_pos[:, 0], all_stim_pos[:, 1]))

            all_stim_pos = all_stim_pos[sorted_indices,:]
            mean_diff = mean_diff[sorted_indices,:,:]

            mean_diff = mean_diff - np.min(np.ravel(mean_diff))

            # dynamically find the pupil position with max over frames
            max_frame = np.max(mean_diff,axis=0)
            max_frame = max_frame - np.min(mean_diff[:])
            max_frame = max_frame.astype(np.uint8)
            thres = np.percentile(np.ravel(max_frame), 99.9)
            # threshold the images to find valid points
            ret, binary = cv2.threshold(max_frame, thres, 1, cv2.THRESH_BINARY)
            # Calculate the moments of the binary image
            M = cv2.moments(binary)
            # Calculate the x and y coordinates of the center of mass
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            # Print the coordinates of the center of mass
            print('Center of mass coordinates: ({}, {})'.format(cx, cy))
            # Crop the frame to be a 200 x 200 centred on the estimated eye position
            mean_diff_cropped = mean_diff[:,cy-100:cy+100,cx-100:cx+100]
            # for each cropped frame check if it has a stim reflection
            # set threshold for reflection
            ref_thres = np.percentile(np.ravel(mean_diff), 99.9)
            all_pos_data = np.empty((0,4),float)
            # allocate space to store the reflection images as list
            all_reflection_frames = []

            for iStim in range(num_stims):
                stim_frame = mean_diff_cropped[iStim,:,:].astype(np.uint8)
                ret, thresh = cv2.threshold(stim_frame, ref_thres, 255, cv2.THRESH_BINARY)
                contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours)>0:
                    # plt.imshow(stim_frame)
                    # plt.show()
                    # plt.imshow(thresh)
                    # plt.show()
                    # if a reflection is found
                    # Compute the mean color (brightness) of each contour
                    mean_colors = []
                    for contour in contours:
                        mask = np.zeros_like(stim_frame)
                        cv2.drawContours(mask, [contour], 0, 255, -1)
                        mean_color = cv2.mean(stim_frame, mask=mask)[:3]
                        mean_colors.append(np.mean(mean_color))
                    # Get the index of the contour with the highest mean color value
                    largest_index = np.argmax(mean_colors)
                    largest_contour = contours[largest_index]
                    # add the midpoint of this island of pixels as a data point 
                    M = cv2.moments(largest_contour)
                    if M['m00'] > 0:
                        # make sure it is a valid contour
                        cx2 = int(M['m10'] / M['m00'])
                        cy2 = int(M['m01'] / M['m00'])
                        all_pos_data = np.append(all_pos_data,np.array([[cx2,cy2,all_stim_pos[iStim,0],all_stim_pos[iStim,1]]]),axis=0)
                        # store reflection frame and randomly colour code
                        stim_frame_normed = stim_frame - np.min(stim_frame)
                        stim_frame_normed = stim_frame_normed / np.max(stim_frame_normed)
                        stim_frame_coloured = np.stack((stim_frame_normed*np.random.rand(),stim_frame_normed*np.random.rand(),stim_frame_normed*np.random.rand()),axis=2)
                        all_reflection_frames.append(stim_frame_coloured)
                        #plt.imshow(all_reflection_frames[-1])
                        #plt.show()

            start_time = time.time()
            # show all reflection frames
            plot_size = int(np.ceil(np.sqrt((len(all_reflection_frames)+2))))
            fig, axs = plt.subplots(plot_size, plot_size)
            for iFrame in range(len(all_reflection_frames)):
                axs[iFrame//plot_size,iFrame%plot_size].imshow(all_reflection_frames[iFrame])
                axs[iFrame//plot_size,iFrame%plot_size].axis('off')
            mean_pretrial_all = np.mean(mean_pretrial,axis=0)
            mean_pretrial_all = mean_pretrial_all[cy-100:cy+100,cx-100:cx+100]
            axs[(iFrame+1)//plot_size,(iFrame+1)%plot_size].imshow(mean_pretrial_all)
            axs[(iFrame+1)//plot_size,(iFrame+1)%plot_size].scatter(all_pos_data[:,0], all_pos_data[:,1], c='r', marker='o', edgecolors='w')
            axs[(iFrame+1)//plot_size,(iFrame+1)%plot_size].axis('off')

            # all_pos_data rows are stim positions and columns are [reflection x, reflection y, stim pos x, stim pos y]
            x=0
            # do fit
            # Fit the polynomial function to the data
            popt_azimuth, _ = curve_fit(polynomial_2D, (all_pos_data[:, 0], all_pos_data[:, 1]), all_pos_data[:, 2])
            popt_elevation, _ = curve_fit(polynomial_2D, (all_pos_data[:, 0], all_pos_data[:, 1]), all_pos_data[:, 3])

            def pixel_to_angle(x, y):
                azimuth = polynomial_2D((x, y), *popt_azimuth)
                elevation = polynomial_2D((x, y), *popt_elevation)
                return azimuth, elevation

            # Assuming the frame has dimensions (height, width)
            height = 200
            width = 200

            # Create a grid of x and y coordinates
            x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))

            def pixel_to_angle_vectorized(x_grid, y_grid):
                azimuth = polynomial_2D((x_grid, y_grid), *popt_azimuth)
                elevation = polynomial_2D((x_grid, y_grid), *popt_elevation)
                return azimuth, elevation

            # Apply the pixel_to_angle_vectorized function to the entire grid
            azimuth_matrix, elevation_matrix = pixel_to_angle_vectorized(x_coords, y_coords)
            x = 0

            axs[(iFrame+2)//plot_size,(iFrame+2)%plot_size].imshow(mean_pretrial_all)
            cont1 = axs[(iFrame+2)//plot_size,(iFrame+2)%plot_size].contour(azimuth_matrix, 50, cmap='viridis', origin='lower')
            axs[(iFrame+2)//plot_size,(iFrame+2)%plot_size].axis('off')
            fig.colorbar(cont1, ax=axs[(iFrame+2)//plot_size,(iFrame+2)%plot_size])

            axs[(iFrame+3)//plot_size,(iFrame+3)%plot_size].imshow(mean_pretrial_all)
            cont2 = axs[(iFrame+3)//plot_size,(iFrame+3)%plot_size].contour(elevation_matrix, 50, cmap='viridis', origin='lower')
            axs[(iFrame+3)//plot_size,(iFrame+3)%plot_size].axis('off')
            fig.colorbar(cont2, ax=axs[(iFrame+3)//plot_size,(iFrame+3)%plot_size])

            # hide any extra plots
            for iPlot in range(iFrame+4,plot_size**2):
                axs[iPlot//plot_size,iPlot%plot_size].axis('off')

            # plt.show()

            print(f"Fit and plot time: {time.time() - start_time:.2f} seconds")

            # inlay the azimuth and elevation maps back into the full frame so they can 
            # be used as a lookup table
            if iCap == 0:
                # store left eye info
                left_map = {}
                left_map['azimuth'] = np.zeros(frame_size)
                left_map['azimuth'][cy-100:cy+100,cx-100:cx+100] = azimuth_matrix
                left_map['elevation'] = np.zeros(frame_size)
                left_map['elevation'][cy-100:cy+100,cx-100:cx+100] = elevation_matrix
                plt.savefig(os.path.join(meta_data_path,'left_eye_calibration_output.png'), dpi=300, bbox_inches='tight')
                plt.clf()
            else:
                # store right eye info
                right_map = {}
                right_map['azimuth'] = np.zeros(frame_size)
                right_map['azimuth'][cy-100:cy+100,cx-100:cx+100] = azimuth_matrix
                right_map['elevation'] = np.zeros(frame_size)
                right_map['elevation'][cy-100:cy+100,cx-100:cx+100] = elevation_matrix
                plt.savefig(os.path.join(meta_data_path,'right_eye_calibration_output.png'), dpi=300, bbox_inches='tight')   
                plt.clf()

        # combine the two eye maps into one nested dict
        pupil_maps  = {}
        pupil_maps['left_map'] = left_map
        pupil_maps['right_map'] = right_map
        # save for later use
        print('Saving pixel->angle mapping map')
        with open(os.path.join(meta_data_path,'eye_pix_angle_map.pickle'), 'wb') as f: pickle.dump(pupil_maps, f)   

# for debugging:
def main():
    x = 0
    # userID = 'adamranson'
    # expID = ['2023-04-18_01_ESMT124','2023-04-18_01_ESMT124']
    # # expID = ['2023-02-24_02_ESMT116'] # lights off eye pos calib
    # #expID = ['2023-04-18_01_ESMT124']
    # #expID = ['2023-02-24_01_ESMT116']
    # run_pupil_calib(userID, expID)

if __name__ == "__main__":
    main()
