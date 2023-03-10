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
all_trials.to_csv(os.path.join(exp_dir_processed, expID + '_all_trials.csv'), index=False)

###########################################################
######## Process S2P data #################################
###########################################################
# check suite2p folder exists to be processed
print('Starting S2P section...')
if os.path.exists(os.path.join(exp_dir_processed, 'suite2p')) and not skip_ca:
    doMerge = False
    resampleFreq = 30
    neuropilWeight = 0.7
    # initiate these as dict:
    alldF = {}
    allF = {}
    allSpikes = {}
    allDepths = {}
    allRoiPix = {}
    allRoiMaps = {}
    allFOV = {}
    
    # this will be used to make all recordings 2 secs shorter than the
    # first ca trace processed to ensure all chs and depths are the same length
    expFrameLength = []
    
    # outputTimes = Timeline.time(1):1/resampleFreq:Timeline.time(end);
    
    # check number of channels
    if os.path.exists(os.path.join(exp_dir_processed, 'ch2')):
        # then there are 2 functional channels
        dataPath = [os.path.join(exp_dir_processed, 'suite2p'),
                    os.path.join(exp_dir_processed, 'ch2', 'suite2p')]
    else:
        dataPath = [os.path.join(exp_dir_processed, 'suite2p')]
    
    # check number of depths
    depthCount = len([d for d in os.listdir(dataPath[0]) if "plane" in d])
    
    if depthCount == 1:
        # then we might be doing frame averaging
        #load(fullfile(expRoot,'tifHeader.mat'));
        #acqNumAveragedFrames = header.acqNumAveragedFrames;
        acqNumAveragedFrames = 1
    else:
        # then we assume no averaging
        acqNumAveragedFrames = 1
    
    # determine which channel has frame timing pulses
    neuralFramesIdx = np.where(np.isin(tl_chNames, 'MicroscopeFrames'))[0][0]
    neuralFramesPulses = np.squeeze((tl_daqData[:,neuralFramesIdx]>1).astype(int))
    # divide the frame counter by the number of depths & averaging factor.
    #Timeline.rawDAQData(:,neuralFramesIdx)=ceil(Timeline.rawDAQData(:,neuralFramesIdx)/depthCount/acqNumAveragedFrames);

    # determine time of each frame
    frameTimes = np.squeeze(tl_time)[np.where(np.diff(neuralFramesPulses)==1)[0]]
    framePulsesPerDepth = len(frameTimes)/depthCount
    frameRate = 1/np.median(np.diff(frameTimes))
    frameRatePerPlane = frameRate/depthCount
    
    # determine timeline times when we want the Ca signal of each cell
    # +1 and -1 are because we want to make sure we only include frame
    # times which are available at all depths
    outputTimes = np.arange(frameTimes[0]+1, frameTimes[-1]-1, 1/resampleFreq)
    
    for iCh in range(len(dataPath)):

        for iDepth in range(depthCount):
            allRoiPix[iCh] = {}
            #allRoiPix[iCh][iDepth] = np.array([])
            allRoiMaps[iCh] = {}
            #allRoiMaps[iCh][iDepth] = np.array([])
            allFOV[iCh] = {}
            # load s2p data
            # check if the big npy file exists and if so load that (this is the non-trucated file, where as the truncated one is the one used for local curation using suite2p)
            if os.path.exists(os.path.join(dataPath[iCh], 'plane'+str(iDepth), 'F_big.npy')):
                Fall = np.load(os.path.join(dataPath[iCh], 'plane'+str(iDepth), 'F_big.npy'))
                Fneu = np.load(os.path.join(dataPath[iCh], 'plane'+str(iDepth), 'Fneu_big.npy'))
                spks = np.load(os.path.join(dataPath[iCh], 'plane'+str(iDepth), 'spks_big.npy'))
            else:
                Fall = np.load(os.path.join(dataPath[iCh], 'plane'+str(iDepth), 'F.npy'))
                Fneu = np.load(os.path.join(dataPath[iCh], 'plane'+str(iDepth), 'Fneu.npy'))
                spks = np.load(os.path.join(dataPath[iCh], 'plane'+str(iDepth), 'spks.npy'))

            s2p_stat = np.load(os.path.join(dataPath[iCh], 'plane'+str(iDepth), 'stat.npy'), allow_pickle=True)
            s2p_ops = np.load(os.path.join(dataPath[iCh], 'plane'+str(iDepth), 'ops.npy'), allow_pickle=True).item()

            # check for mismatch between frames trigs and frames in tiff
            if abs(framePulsesPerDepth-Fall.shape[1])/max([framePulsesPerDepth,Fall.shape[1]]) > 0.01:
                pcDiff = round(abs(framePulsesPerDepth-Fall.shape[1])/max([framePulsesPerDepth,Fall.shape[1]]) * 100)
                raise Exception('There is a mismatch between between frames trigs and frames in tiff - ' + str(pcDiff) + '% difference')
            # load numpy file containing cell classification
            cellValid = np.load(os.path.join(dataPath[iCh], 'plane'+str(iDepth), 'iscell.npy'))
            cellValid = cellValid[:,0]
            # overall video contamination subtraction

            if Fneu.shape[0] == 0:
                # then there are no rois detected in the plane
                meanFrameTimecourse = np.zeros([1,Fall.shape[0]])
            else:
                # RECALC this using median of all neuropils
                meanFrameTimecourse = np.median(Fneu,0)
                meanFrameTimecourse = meanFrameTimecourse - min(meanFrameTimecourse)
            # check if any roi Fs are all zero. s2p sometimes throws these
            # up for some reason. if these are found set iscell to false
            zeroROIs = ((np.max(Fall, axis=1) == 0) & (np.min(Fall, axis=1) == 0)).astype(int)

            if np.sum(zeroROIs) > 0:
                print('Warning: '+str(np.sum(zeroROIs))+' zero flat lined rois...')
                cellValid[np.where(zeroROIs==1)] = 0

            # find cells which are part of merges and set iscell to 0
            # this refers to merges in suite2p
            totalMerges = 0
            for iCell in range(len(s2p_stat)):
                if 'ismerge' in s2p_stat[iCell]:
                    if s2p_stat[iCell]['inmerge'] == 1:
                        # then the cell is included in a merged roi
                        cellValid[iCell] = 0
                        totalMerges = totalMerges + 1

            if totalMerges > 0:
                print(f"Merges found: {totalMerges}")

            # remove cells with iscell = 0 but keep record of original
            # suite2p output cell numbers
            # this is the valid neuropil / F / Spks
            Fneu_valid = np.squeeze(Fneu[np.where(cellValid==1), :])
            F_valid = np.squeeze(Fall[np.where(cellValid==1), :])
            Spks_valid = np.squeeze(spks[np.where(cellValid==1), :])
            s2pIndices = np.where(cellValid==1)
            xpix, ypix = [], []
            validCellIDs = np.where(cellValid==1)[0]
            # this is to later store the x and ypix of each valid cell
            for iCell in range(len(validCellIDs)):
                currentCell = validCellIDs[iCell]
                xpix.append(s2p_stat[currentCell]['xpix'])
                ypix.append(s2p_stat[currentCell]['ypix'])

            # remove potential stimulus artifact - i.e. mean of frame which
            # is extracted above
            Fneu_valid = Fneu_valid - np.tile(meanFrameTimecourse, (Fneu_valid.shape[0], 1))
            F_valid = F_valid - np.tile(meanFrameTimecourse, (F_valid.shape[0], 1))

            # neuropil subtraction
            F_valid = F_valid - (Fneu_valid * neuropilWeight)

            # Ensure min(corrected F) > 10
            FMins = np.min(F_valid, axis=1)
            plt.subplot(1, 2, 1)
            plt.hist(FMins)
            plt.title(['Distribution of original', 'F values of ROIS'])

            if np.min(FMins) < 20:
                print('Frame mean and neuropil subtraction give ROIs with F < 20')
                print('Offsetting all F by', (np.min(FMins)*-1)+20)
                F_valid = F_valid + (np.min(FMins)*-1)+20
                
            FMins = np.min(F_valid, axis=1)
            plt.subplot(1, 2, 2)
            plt.hist(FMins)
            plt.title(['Distribution of F values', 'of ROIS after forcing > 20'])
            plt.draw()

            # do merge was removed from here - this was bill's code for merging correlated rois
            # what to do if not merging
            # make a roi map for the depth that can be used for longitudinal imaging etc
            roiPix = []
            # make a blank roi map
            roiMap = np.zeros(np.shape(s2p_ops['meanImg']))
            for iRoi in range(F_valid.shape[0]):
                # collect pix in ROI
                roiPix.append(np.ravel_multi_index((ypix[iRoi]+1,xpix[iRoi]+1), np.shape(s2p_ops['meanImg'])))
                # label ROI map
                roiMap[ypix[iRoi], xpix[iRoi]] = iRoi+1
            plt.figure
            plt.imshow(roiMap), plt.show()
            # crop F down to above established max frames
            # F = F[:, :expFrameLength]

            # dF/F calculation
            # this window should equate to 10 secs
            smoothingWindowSize = (3 * frameRatePerPlane).round().astype(int)
            baseline_min_window_size = (10 * frameRatePerPlane).round().astype(int)
            kernel = np.ones((1, smoothingWindowSize)) / smoothingWindowSize
            smoothed = signal.convolve2d(F_valid, kernel, mode='same')
            # remove edge effects
            # at start
            smoothed[:, :smoothingWindowSize] = np.tile(smoothed[:, smoothingWindowSize+1].reshape(smoothed.shape[0],1), [1, smoothingWindowSize])
            smoothed[:, -smoothingWindowSize:] = np.tile(smoothed[:, -smoothingWindowSize-1].reshape(smoothed.shape[0],1), [1, smoothingWindowSize])
            # replace nans with large values (so they don't get picked up as mins)
            smoothed[np.isnan(smoothed)] = np.max(smoothed)*2
            # sliding min:
            # compute the sliding window minimum with a window size of 3 over the second dimension
            baseline = np.apply_along_axis(lambda smoothed: np.minimum.accumulate(np.concatenate([smoothed, np.repeat(smoothed[-1], baseline_min_window_size)]))[baseline_min_window_size:], axis=1, arr=smoothed)

            # for iCell in range(10,20):
            #     cell_to_plot = iCell
            #     plt.figure(),plt.plot(F_valid[cell_to_plot,:]),plt.plot(min_x[cell_to_plot,:]), plt.show
            #     plt.figure(),plt.imshow(dF, aspect='auto'),plt.colorbar(),plt.show()

            # # calculate dF/F
            dF = (F_valid-baseline) / baseline
            # get times of each frame
            depthFrameTimes = frameTimes[iDepth+1:len(frameTimes):depthCount]
            # make sure there are not more times than frames
            depthFrameTimes = depthFrameTimes[:dF.shape[1]]
            # resample to get desired sampling rate
            dF_resampled = interpolate.interp1d(depthFrameTimes, dF.T, axis=0, fill_value="extrapolate")(outputTimes).T
            F_resampled = interpolate.interp1d(depthFrameTimes, F_valid.T, axis=0, fill_value="extrapolate")(outputTimes).T
            Spks_resampled = interpolate.interp1d(depthFrameTimes, Spks_valid.T, axis=0, fill_value="extrapolate")(outputTimes).T
            # if there is only one cell ensure it is rotated to (cell,time) orientation
            if len(dF_resampled.shape) == 1:
                # add the new axis to make sure it is (cell,time)
                dF_resampled = dF_resampled[np.newaxis,:]
                F_resampled = F_resampled[np.newaxis,:]
                Spks_resampled = Spks_resampled[np.newaxis,:]

            # pick out valid cells
            allRoiPix[iCh] = {}
            
            if dF_resampled.shape[0] > 0:
                # then we have some rois
                if not(iCh in alldF):
                    # if these are the first cells being added then initiate the array in the dict item with the right size
                    alldF[iCh] = dF_resampled
                    allF[iCh] = F_resampled
                    allSpikes[iCh] = Spks_resampled
                    allDepths[iCh] = np.tile(iDepth, (np.sum(cellValid[:]).astype(int), 1))
                else:
                    # concatenate to what is already there
                    alldF[iCh] = np.concatenate((alldF[iCh],dF_resampled),axis=0)
                    allF[iCh] = np.concatenate((allF[iCh],F_resampled),axis=0)
                    allSpikes[iCh] = np.concatenate((allSpikes[iCh],Spks_resampled),axis=0)
                    allDepths[iCh] = np.concatenate([allDepths[iCh], np.tile(iDepth, (np.sum(cellValid[:]).astype(int), 1))],axis=0)

            allRoiPix[iCh][iDepth] = roiPix
            allRoiMaps[iCh][iDepth] = roiMap
            allFOV[iCh][iDepth] = s2p_ops['meanImg']

    print('Saving 2-photon data...')

    # save as CSV
    for iCh in range(len(alldF)):
        # make a dict where all of the experiment data is stored
        ca_data = {}
        ca_data['dF']           = alldF[iCh]
        ca_data['F']            = allF[iCh]
        ca_data['Spikes']       = allSpikes[iCh]
        ca_data['Depths']       = allDepths[iCh]
        ca_data['AllRoiPix']    = allRoiPix[iCh]
        ca_data['AllRoiMaps']   = allRoiMaps[iCh]
        ca_data['AllFOV']       = allFOV[iCh]
        output_filename = 's2p_ch' + str(iCh)+'.pickle'
        pickle_out = open(os.path.join(exp_dir_processed_recordings,output_filename),"wb")
        pickle.dump(ca_data, pickle_out)
        pickle_out.close()
        # pickle_in = open(os.path.join(exp_dir_processed_recordings,output_filename),"rb")
        # example_dict = pickle.load(pickle_in)

#################################
#### ePhys data #################
#################################
ePhys1Idx = np.where(np.isin(tl_chNames, 'EPhys1'))
ePhys2Idx = np.where(np.isin(tl_chNames, 'EPhys2'))
ePhys1Data = np.squeeze(tl_daqData[:, ePhys1Idx])[np.newaxis,:]
ePhys2Data = np.squeeze(tl_daqData[:, ePhys2Idx])[np.newaxis,:]
ephys_combined = np.concatenate((tl_time,ePhys1Data,ePhys2Data),axis=0)
np.save(os.path.join(exp_dir_processed_recordings,'ephys.npy'),ephys_combined)

# ePhys1Idx  = find(ismember(Timeline.chNames,'EPhys1'));
# ePhys2Idx  = find(ismember(Timeline.chNames,'EPhys2'));
# writematrix([Timeline.time',Timeline.daqData(:,[ePhys1Idx ePhys2Idx])],fullfile(recordingsRoot,'ephys.csv'));