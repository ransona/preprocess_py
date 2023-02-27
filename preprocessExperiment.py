import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
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


# Process Bonsai stuff
frame_events = pd.read_csv(os.path.join(exp_dir, expID + '_FrameEvents.csv'), names=['Frame', 'Timestamp', 'Sync', 'Trial'],
                           header=None, dtype={'Frame':np.float32, 'Timestamp':np.float32, 'Sync':np.float32, 'Trial':np.float32})

# Find BV times when digital flips
Timestamp = frame_events['Timestamp'].values
Sync = frame_events['Sync'].values
flip_idx = np.where(np.diff(Sync) == -1)
flip_times_bv = frame_events['Timestamp'][np.where((np.diff(frame_events['Sync']) == -1))[0] + 1]
x = frame_events['Timestamp']

plt.plot(x.values)

# Find TL times when digital flips
bv_ch = np.where(np.isin(Timeline.chNames, 'Bonvision'))[0][0]
tl_dig_thresholded = Timeline.daqData[:, bv_ch] > 2.5
flip_times_tl = Timeline.time[np.where((np.diff(tl_dig_thresholded) == -1))[0] + 1]

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
mdl1 = smf.ols('flip_times_tl ~ flip_times_bv',
               data=pd.DataFrame({'flip_times_bv': flip_times_bv, 'flip_times_tl': flip_times_tl})).fit()
flip_times_bv_predicted = mdl1.predict(pd.DataFrame({'flip_times_bv': flip_times_bv}))

# get trial onset times
# in BV time
trialOnsetTimesBV = [FrameEvents.Timestamp[0], FrameEvents.Timestamp[np.where(np.diff(FrameEvents.Trial) == 1)[0] + 1]]
# in TL time
trialOnsetTimesTL = mdl1.predict(trialOnsetTimesBV)

paramNames_gratings = ['stimnumber', 'featurenumber', 'featuretype', 'angle', 'size', 'x', 'y', 'contrast', 'opacity',
                       'phase', 'freq', 'speed', 'dcycle', 'onset', 'duration']
paramNames_video = ['stimnumber', 'featurenumber', 'featuretype', 'angle', 'width', 'height', 'x', 'y', 'loop', 'speed',
                    'name', 'onset', 'duration']

# make a matrix for csv output of trial onset time and trial stimulus type
# check number of trial onsets matches between bonvision and bGUI
if len(trialOnsetTimesTL) != len(expDat['stimOrder']):
    raise ValueError(
        'Number of trial onsets doesn\'t match between bonvision and bGUI - there is a likely logging issue')
trialTimeMatrix = np.column_stack((trialOnsetTimesTL, expDat['stimOrder']))

# store the params of each stim conditions in a csv
allStimTypes_mat = []

for iStimType in range(len(expDat['stims'])):
    # within each stim are features, i.e. gratings or movies
    for iFeature in range(len(expDat['stims'][iStimType]['features'])):
        allStimTypes_mat.append([iStimType + 1, iFeature + 1])
        # cycle through parameters of stim feature
        if expDat['stims'][iStimType]['features'][iFeature]['name'][0] == 'grating':
            param_list = paramNames_gratings[3:]
            allStimTypes_mat[-1].append('0')
        elif expDat['stims'][iStimType]['features'][iFeature]['name'][0] == 'movie':
            param_list = paramNames_video[3:]
            allStimTypes_mat[-1].append('1')
        for iParam in range(len(param_list)):
            if param_list[iParam] == 'size':
                param_number = \
                np.where(np.array(expDat['stims'][iStimType]['features'][iFeature]['params']) == 'width')[0][0]
            else:
                param_number = \
                np.where(np.array(expDat['stims'][iStimType]['features'][iFeature]['params']) == param_list[iParam])[0][
                    0]
            allStimTypes_mat[-1].append(expDat['stims'][iStimType]['features'][iFeature]['vals'][param_number])

# Add running trace
Encoder = pd.read_csv(os.path.join(expRootMeta, expID + '_Encoder.csv'))
Encoder.columns = ['Frame', 'Timestamp', 'Trial', 'Position']
wheelPos = Encoder.Position.values
wheelTimestamps = mdl1.predict(Encoder.Timestamp.values)
# Resample wheel to linear timescale
wheelLinearTimescale = np.arange(wheelTimestamps[0], wheelTimestamps[-1], 0.01)
f = interpolate.interp1d(wheelTimestamps, wheelPos, kind='linear')
wheelPos2 = pd.Series(f(wheelLinearTimescale)).rolling(window=50, center=True).mean().fillna(method='ffill').fillna(
    method='bfill')
wheelSpeed = ((np.insert(np.diff(wheelPos2.values) * -1, 0, 0) * (62 / 1024)) * 100)

# Save data
bvDataRoot = os.path.join(expRootLocal, 'bonsai')
if not os.path.exists(bvDataRoot):
    os.mkdir(bvDataRoot)
np.savetxt(os.path.join(recordingsRoot, 'WheelPos.csv'), np.column_stack((wheelLinearTimescale, wheelPos2)),
           delimiter=',')
np.savetxt(os.path.join(recordingsRoot, 'WheelSpeed.csv'), np.column_stack((wheelLinearTimescale, wheelSpeed)),
           delimiter=',')
np.savetxt(os.path.join(bvDataRoot, 'Trials.csv'), trialTimeMatrix, delimiter=',')


