import os
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from scipy import interpolate
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy import signal
from scipy import ndimage
import matplotlib.pyplot as plt
import pickle

# check if running local or colab
try:
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive/mydrive')
    drive_prefix = '/content/drive/mydrive'
else:
    # set path appropriately locally
    drive_prefix = os.path.join('g:\\', 'My Drive')
    # this will be used to resolve gdrive shortcuts:
    import win32com.client
    shell = win32com.client.Dispatch("WScript.Shell")

# expID
expID = '2022-10-19_08_ESMT101'
# user ID to use to place processed data
userID = 'AR_RRP'
# get animal ID from experiment ID
animalID = expID[14:]
# path to root of raw data
remote_repository_root = os.path.join(drive_prefix,'Remote_Repository')
# path to root of processed data
processed_root = os.path.join(drive_prefix,'Remote_Repository_Processed',userID)

skip_ca = False

# resolve gdrive shortcuts if needed
if os.path.exists(remote_repository_root+'.lnk'):
    shortcut = shell.CreateShortCut(remote_repository_root+'.lnk')
    remote_repository_root = shortcut.Targetpath
if os.path.exists(processed_root+'.lnk'):
    shortcut = shell.CreateShortCut(processed_root+'.lnk')
    processed_root = shortcut.Targetpath

# complete path to processed experiment data
exp_dir_processed = os.path.join(processed_root, animalID, expID)
# complete path to processed experiment data recordings
exp_dir_processed_recordings = os.path.join(processed_root, animalID, expID,'recordings')
# complete path to raw experiment data
exp_dir = os.path.join(remote_repository_root, animalID, expID)

animalID = expID[14:]

if not os.path.exists(exp_dir_processed_recordings):
    os.mkdir(exp_dir_processed_recordings)

# is Timeline file is present then assume all other meta data files are
# also on the local disk. if not then assume they all need to be loaded
# from the server. this will only work when running this on the UAB network
# where the meta data files can be accessed.
if os.path.exists(os.path.join(exp_dir, expID + '_Timeline.mat')):
    # assume all meta data is available in the expRoot folder.
    Timeline = loadmat(os.path.join(exp_dir, expID + '_Timeline.mat'))
    Timeline = Timeline['timelineSession']
    print('Meta data found in Remote_Repository')

# load the stimulus parameter file produced by matlab by the bGUI
# this includes stim parameters and stimulus order
try:
    stim_params = loadmat(os.path.join(exp_dir, expID + '_stim.mat'))
except:
    raise Exception('Stimulus parameter file not found - this experiment was probably from pre-Dec 2021.')

# Do some plots to help spot dodgy data
# fig, axs = plt.subplots(len(Timeline['chNames']), 1)
# if Timeline['daqData'].shape[0] > 60000:
#     samplesToPlot = range(60000)
# else:
#     samplesToPlot = range(Timeline['daqData'].shape[0])
# for iPlot in range(len(Timeline['chNames'])):
#     axs[iPlot].plot([x / 1000 for x in samplesToPlot], Timeline['daqData'][samplesToPlot, iPlot])
#     axs[iPlot].set_title(Timeline['chNames'][iPlot])
# plt.subplots_adjust(hspace=0.5)
# plt.show()

# get timeline file in a usable format after importing to python
tl_chNames = Timeline['chNames'][0][0][0][0:]
tl_daqData = Timeline['daqData'][0,0]
tl_time    = Timeline['time'][0][0]

###################################
###################################
###################################

# Load the data from the file
eyeMeta1_mat = os.path.join(exp_dir,(expID + "_eyeMeta1.mat"))
eyeMeta1 = loadmat(eyeMeta1_mat)

eTrackDataML = eyeMeta1["eTrackData"]
eTrackData = {}
eTrackData['FrameTimes'] = eTrackDataML['frameTimes'][0][0][0]
eTrackData['FrameCount'] = eTrackDataML['frameCount'][0][0][0][0]
eTrackData['PosPulseTimes'] = eTrackDataML['posPulseTimes'][0][0]
eTrackData['negPulseTimes'] = eTrackDataML['negPulseTimes'][0][0]
eTrackData['StartTime'] = eTrackDataML['startTime'][0][0][0][0]

# Find the index of the camera channel
camIdx = np.where(np.isin(tl_chNames, 'EyeCamera'))[0][0]

# Get the camera pulse trace and frame pulse times
camPulseTrace = (tl_daqData[:, camIdx] > 2.5).astype(int)
framePulseTimes = tl_time[0,np.where(np.diff(camPulseTrace) == 1)][0]

# Perform a quality check on the frame pulse times
if np.min(np.diff(framePulseTimes)) < 16:
    plt.figure()
    if tl_time.shape[1] > 100000:
        plt.plot(tl_time[0,:100000], tl_daqData[:100000, camIdx])
    else:
        plt.plot(tl_time[0,:], tl_daqData[:, camIdx])
    plt.title("Eye camera timing pulses (ch{camIdx} of DAQ)")
    plt.xlabel("Time (secs)")
    plt.ylabel("Voltage (volts)")
    print("The timing pulses on the eye camera look faulty - see the figure")
    raise ValueError("The timing pulses on the eye camera look faulty - see the figure")

# Adjust the logged frame times to be approximately in timeline time
loggedFrameTimes = eTrackData["FrameTimes"] - eTrackData["FrameTimes"][0]
loggedFrameTimes = loggedFrameTimes + framePulseTimes[0]

# Periodically correct the logged frame times to the timeline clock
framePulseFrameNumbers = np.arange(0, len(framePulseTimes)*200, 200)

for iPulse in range(len(framePulseTimes)):
    # at each pulse calculate how much the systems have gone out of
    # sync and correct the next 200 frame times in loggedFrameTimes
    tlTimeOfPulse = framePulseTimes[iPulse]
    eyecamTimeOfPulse = loggedFrameTimes[framePulseFrameNumbers[iPulse]]
    driftAtPulse = tlTimeOfPulse - eyecamTimeOfPulse
    # corrected logged times
    if iPulse < len(framePulseTimes)-1:
        loggedFrameTimes[framePulseFrameNumbers[iPulse]:framePulseFrameNumbers[iPulse]+200] = \
            loggedFrameTimes[framePulseFrameNumbers[iPulse]:framePulseFrameNumbers[iPulse]+200] + driftAtPulse
    else:
        loggedFrameTimes[framePulseFrameNumbers[iPulse]:] = \
            loggedFrameTimes[framePulseFrameNumbers[iPulse]:] + driftAtPulse

# define frames we want to know the times of
allFrameNumbers = np.arange(1, eTrackData["FrameCount"]+1)
allFrameTimes = np.interp(allFrameNumbers, framePulseFrameNumbers, framePulseTimes)

# debugging plots:
# fig, (ax1, ax2) = plt.subplots(1, 2)
# ax1.plot(allFrameTimes, allFrameNumbers)
# ax1.scatter(framePulseTimes, framePulseFrameNumbers, color='red')
# ax1.plot(eTrackData.frameTimes+allFrameTimes[0], allFrameNumbers)
# ax2.plot(allFrameTimes[:5000], allFrameTimes[:5000]-eTrackData.frameTimes[:5000])

frameRate = 1/np.median(np.diff(loggedFrameTimes))
print(f"Detected eye cam frame rate = {frameRate} Hz")
np.savetxt(os.path.join(recordingsRoot, 'eyeFrames.csv'), np.column_stack((loggedFrameTimes, allFrameNumbers)), delimiter=',')

# store detected eye details with timeline timestamps
# load
import pandas as pd
import numpy as np
import os

try:
    left_eye_data = np.load(os.path.join(expRootLocal, 'dlcEyeLeft.npy'), allow_pickle=True).item()
    left_eye_data = left_eye_data['eyeDat']
    right_eye_data = np.load(os.path.join(expRootLocal, 'dlcEyeRight.npy'), allow_pickle=True).item()
    right_eye_data = right_eye_data['eyeDat']
    
    # resample to 10Hz constant rate
    new_time_vector = np.arange(logged_frame_times[0], logged_frame_times[-1], 0.1)
    left_table = pd.DataFrame(columns=['time', 'x', 'y', 'radius', 'velocity', 'qc'])
    left_table['time'] = new_time_vector
    left_table['x'] = np.interp(new_time_vector, logged_frame_times, left_eye_data['x'])
    left_table['y'] = np.interp(new_time_vector, logged_frame_times, left_eye_data['y'])
    left_table['radius'] = np.interp(new_time_vector, logged_frame_times, left_eye_data['radius'])
    left_table['velocity'] = np.interp(new_time_vector, logged_frame_times, left_eye_data['velocity'])
    left_table['qc'] = np.interp(new_time_vector, logged_frame_times, left_eye_data['qc'])
    left_table.to_csv(os.path.join(recordingsRoot, 'left_eye.csv'), index=False)

    right_table = pd.DataFrame(columns=['time', 'x', 'y', 'radius', 'velocity', 'qc'])
    right_table['time'] = new_time_vector
    right_table['x'] = np.interp(new_time_vector, logged_frame_times, right_eye_data['x'])
    right_table['y'] = np.interp(new_time_vector, logged_frame_times, right_eye_data['y'])
    right_table['radius'] = np.interp(new_time_vector, logged_frame_times, right_eye_data['radius'])
    right_table['velocity'] = np.interp(new_time_vector, logged_frame_times, right_eye_data['velocity'])
    right_table['qc'] = np.interp(new_time_vector, logged_frame_times, right_eye_data['qc'])
    right_table.to_csv(os.path.join(recordingsRoot, 'right_eye.csv'), index=False)

except:
    print('Problem loading or processing DLC data')
