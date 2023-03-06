import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy import interpolate
from scipy.io import loadmat
import matplotlib.pyplot as plt

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
# Process Bonsai stuff
frame_events = pd.read_csv(os.path.join(exp_dir, expID + '_FrameEvents.csv'), names=['Frame', 'Timestamp', 'Sync', 'Trial'],
                           header=None, skiprows=[0], dtype={'Frame':np.float32, 'Timestamp':np.float32, 'Sync':np.float32, 'Trial':np.float32})

# Find BV times when digital flips
Timestamp = frame_events['Timestamp'].values
Sync = frame_events['Sync'].values
Trial = frame_events['Trial']
flip_idx = np.where(np.diff(Sync) == -1)
flip_times_bv = np.squeeze(Timestamp[np.where((np.diff(Sync) == -1))[0]])

# plt.plot(np.diff(Sync))
# plt.ion()
# plt.show()
#
# plt.plot(Timestamp)
# plt.ion()
# plt.show()

# Find TL times when digital flips
bv_ch = np.where(np.isin(tl_chNames, 'Bonvision'))
tl_dig_thresholded = np.squeeze((tl_daqData[:, bv_ch] > 2.5).astype(int))

flip_times_tl = np.squeeze(tl_time[0,np.where(np.diff(tl_dig_thresholded) == -1)])
# Check NI DAQ caught as many sync pulses as BV produced
pulse_diff = len(flip_times_tl) - len(flip_times_bv)
print(str(len(flip_times_tl)) + ' pulses found in TL')

if pulse_diff > 0:
    print(str(pulse_diff) + ' more pulses in TL')
    flip_times_tl = flip_times_tl[:len(flip_times_bv)]
elif pulse_diff < 0:
    print(str(pulse_diff * -1) + ' more pulses in BV')
    raise ValueError('Pulse mismatch')
else:
    print('Pulse match')

# Make model to convert BV time to TL time
mdl1 = LinearRegression()
mdl1.fit(flip_times_bv.reshape((-1, 1)), flip_times_tl)

#mdl1 = LinearRegression('flip_times_tl ~ flip_times_bv',
#               data=pd.DataFrame({'flip_times_bv': flip_times_bv, 'flip_times_tl': flip_times_tl})).fit()

#flip_times_bv_predicted = mdl1.predict(pd.DataFrame({'flip_times_bv': flip_times_bv}))

# get trial onset times
# in BV time
# find the moments when the (bonsai) trial number increments
trialOnsetTimesBV = Timestamp[np.where(np.diff(Trial)==1)]
# add in first trial onset
trialOnsetTimesBV = np.insert(trialOnsetTimesBV,0,Timestamp[0])
# in TL time
trialOnsetTimesTL = mdl1.predict(pd.DataFrame(trialOnsetTimesBV))

paramNames_gratings = ['stimnumber', 'featurenumber', 'featuretype', 'angle', 'size', 'x', 'y', 'contrast', 'opacity',
                       'phase', 'freq', 'speed', 'dcycle', 'onset', 'duration']
paramNames_video = ['stimnumber', 'featurenumber', 'featuretype', 'angle', 'width', 'height', 'x', 'y', 'loop', 'speed',
                    'name', 'onset', 'duration']

# load matlab expData file
expData = loadmat(os.path.join(exp_dir, expID + '_stim.mat'))
stims = expData['expDat']['stims']
stims = stims[0][0][0]

#2022-10-19_08_ESMT101_stim_order.csv
stim_info = pd.read_csv(os.path.join(exp_dir, expID + '_stim.csv'))
stim_order = pd.read_csv(os.path.join(exp_dir, expID + '_stim_order.csv'), header=None)

#stims[features][0=feats,1=reps][0][1][vals,params,type][0]
# cycle through each stimulus condition to find which has most features

# make a matrix for csv output of trial onset time and trial stimulus type
# check number of trial onsets matches between bonvision and bGUI
if len(trialOnsetTimesTL) != stim_order.shape[0]:
    raise ValueError(
        'Number of trial onsets doesn\'t match between bonvision and bGUI - there is a likely logging issue')

# make the matrix of trial onset times
trialTimeMatrix = np.column_stack((trialOnsetTimesTL, stim_order.values))

# store the params of each stim conditions in a csv
allStimTypes_mat = []
# Add running trace
Encoder = pd.read_csv(os.path.join(exp_dir, expID + '_Encoder.csv'), names=['Frame', 'Timestamp', 'Trial', 'Position'],
                           header=None, skiprows=[0]) #, dtype={'Frame':np.float32, 'Timestamp':np.float32, 'Trial':np.int64, 'Position':np.int64})
wheelPos = Encoder.Position.values
# deal with wrap around of rotary encoder position
wheelPosDif = np.diff(wheelPos)
wheelPosDif[wheelPosDif > 50000] -= 2**16
wheelPosDif[wheelPosDif < -50000] += 2**16
wheelPos = np.cumsum(wheelPosDif)
wheelPos=np.append(wheelPos,wheelPos[-1])
wheelTimestamps = mdl1.predict(Encoder.Timestamp.values.reshape(-1,1))
# Resample wheel to linear timescale
wheelLinearTimescale = np.arange(wheelTimestamps[0], wheelTimestamps[-1], 0.01)
f = interpolate.interp1d(wheelTimestamps, wheelPos, kind='linear')
wheelPos2 = pd.Series(f(wheelLinearTimescale)).rolling(window=50, center=True).mean().fillna(method='ffill').fillna(
    method='bfill')
wheelSpeed = ((np.insert(np.diff(wheelPos2.values) * -1, 0, 0) * (62 / 1024)) * 100)

# Save data
bvDataRoot = os.path.join(exp_dir_processed, 'bonsai')
if not os.path.exists(bvDataRoot):
    os.mkdir(bvDataRoot)
np.savetxt(os.path.join(exp_dir_processed_recordings, 'WheelPos.csv'), np.column_stack((wheelLinearTimescale, wheelPos2)),
           delimiter=',')
np.savetxt(os.path.join(exp_dir_processed_recordings, 'WheelSpeed.csv'), np.column_stack((wheelLinearTimescale, wheelSpeed)),
           delimiter=',')
# output a csv file which contains dataframe of all trials with first column showing trial onset time

#np.savetxt(os.path.join(bvDataRoot, 'Trials.csv'), trialTimeMatrix, delimiter=',')

# read the all trials file, append trial onset times to first column (trialOnsetTimesTL)
all_trials = pd.read_csv(os.path.join(exp_dir, expID + '_all_trials.csv'))
all_trials.insert(0,'time',trialOnsetTimesTL)
pd.DataFrame.to_csv(os.path.join(exp_dir_processed, expID + '_all_trials.csv'),)
all_trials.to_csv(os.path.join(exp_dir_processed, expID + '_all_trials.csv'), index=False)

# next up process ca signals!