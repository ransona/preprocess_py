import pickle
import os
import organise_paths
import numpy as np
import matplotlib.pyplot as plt

userID = 'adamranson'
expID = '2023-03-01_01_ESMT107'  # <-- this is a stim artifact experiment

# the organise_paths.find_paths(userID, expID) gives you various useful
# paths based on an experiment ID
animalID, remote_repository_root, processed_root, exp_dir_processed, exp_dir_raw = organise_paths.find_paths(userID, expID)
# os.path.join combined strings to make a path that will work on whatever 
# operating system the function is run on
exp_dir_processed_recordings = os.path.join(exp_dir_processed,'recordings')
exp_dir_processed_cut = os.path.join(exp_dir_processed,'cut')

# DEEPLABCUT data:
# load DLC data for an experiment. it is in a pickle file which is basically
# a way to save any python object / data type and preserve its structure
# this is the left eye data:
left_resampled = pickle.load(open(os.path.join(exp_dir_processed_recordings,'dlcEyeLeft_resampled.pickle'), "rb"))
# left_resampled['t']      - the time vector which is at 10Hz
# left_resampled['x']          - x positions
# left_resampled['y']          - y positions
# left_resampled['radius']     - pupil radius
# left_resampled['velocity']   - pupil velocity
# left_resampled['qc']         - quality control, 0 = passed, 1 = failed to detect eye shape, 2 = not enough pupil points detected
# left_resampled['frame']      - the video frame that the data point came from
# plot time vs pupil radius

# # Create a 3-row subplot to plot pupil position, radius and velocity
# fig, axs = plt.subplots(3, 1, figsize=(8, 10))
# # Plot data on each subplot
# axs[0].plot(left_resampled['t'], left_resampled['x']-np.nanmedian(left_resampled['x']))
# axs[0].plot(left_resampled['t'], left_resampled['y']-np.nanmedian(left_resampled['y']))
# axs[0].set_title('Pupil x and y position')
# axs[1].plot(left_resampled['t'], left_resampled['radius'])
# axs[1].set_title('Pupil radius')
# axs[2].plot(left_resampled['t'], left_resampled['velocity'])
# axs[2].set_title('Pupil velocity')
# # Add overall title and adjust spacing
# fig.suptitle('Summary of pupil data')
# fig.tight_layout()
# # Show the plot
# plt.show()

# load the cut eye data
eye_left_cut = pickle.load(open(os.path.join(exp_dir_processed_cut,'eye_left_cut.pickle'), "rb"))
# print the shape of the array
print(eye_left_cut['x'].shape) # <- (trial,time)
# make a plot showing pupil radius during all trials
plt.figure()
# rotate 90 to make it plot each trial as a trace
plt.plot(eye_left_cut['t'],np.rot90(eye_left_cut['radius']))
plt.title('Pupil radius during each trial')
plt.xlabel('Time (s)')
plt.ylabel('Pupil radius (pix)')


# Calcium imaging data
# Load uncut trace
# Ch 0 = green, 1 = red
Ch = 0
ca_data = pickle.load(open(os.path.join(exp_dir_processed_recordings,('s2p_ch' + str(Ch)+'.pickle')), "rb"))
# ca_data['dF'][roi,time]
# ca_data['F'][roi,time]
# ca_data['Spikes'][roi,time]
# ca_data['t'][time]
# plot dF/F of first ROI
# plt.figure()
# plt.plot(ca_data['t'],ca_data['dF'][0,:])
# plt.show()

# load cut traces
s2p_dF_cut = pickle.load(open(os.path.join(exp_dir_processed_cut,'s2p_ch0_dF_cut.pickle'), "rb"))
# make a plot showing dF/F during all trials for roi 0
roi = 0
plt.figure()
plt.plot(s2p_dF_cut['t'],np.rot90(s2p_dF_cut['dF'][roi,:,:]))
plt.title('dF during each trial')
plt.xlabel('Time (s)')
plt.ylabel('dF')