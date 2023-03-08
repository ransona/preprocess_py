import os
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from scipy import interpolate
from scipy import signal
from scipy import ndimage
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

# get timeline file in a usable format after importing to python
tl_chNames = Timeline['chNames'][0][0][0][0:]
tl_daqData = Timeline['daqData'][0,0]
tl_time    = Timeline['time'][0][0]
## THIS IS THE BIT FROM THE COMPLETE PIPELINE SCRIPT ^

# check suite2p folder exists to be processed
if os.path.exists(os.path.join(exp_dir_processed, 'suite2p')) and not skip_ca:
    doMerge = False
    resampleFreq = 30
    neuropilWeight = 0.7
    
    alldF = {}
    allF = {}
    allSpikes = {}
    allDepths = {}
    allRoiPix = {}
    allRoiMaps = {}
    
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
    framePulsesPerDepth = len(frameTimes)/len(dataPath)
    frameRate = 1/np.median(np.diff(frameTimes))
    
    # determine timeline times when we want the Ca signal of each cell
    # +1 and -1 are because we want to make sure we only include frame
    # times which are available at all depths
    outputTimes = np.arange(frameTimes[0]+1, frameTimes[-1]-1, 1/resampleFreq)
    
    for iCh in range(len(dataPath)):
        alldF[iCh] = np.array([])
        allF[iCh] = np.array([])
        allSpikes[iCh] = np.array([])
        allDepths[iCh] = np.array([])

        for iDepth in range(depthCount):
            allRoiPix[iCh] = {}
            allRoiPix[iCh][iDepth] = np.array([])
            allRoiMaps[iCh] = {}
            allRoiMaps[iCh][iDepth] = np.array([])
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
            Spks = np.squeeze(spks[np.where(cellValid==1), :])
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
            # this window should equate to 100 secs
            smoothingWindowSize = 100 * resampleFreq
            kernel = np.ones((1, smoothingWindowSize)) / smoothingWindowSize
            smoothed = signal.convolve2d(F_valid, kernel, mode='same')
            # remove edge effects
            # at start
            smoothed[:, :smoothingWindowSize] = np.tile(smoothed[:, smoothingWindowSize+1].reshape(smoothed.shape[0],1), [1, smoothingWindowSize])
            smoothed[:, -smoothingWindowSize:] = np.tile(smoothed[:, -smoothingWindowSize-1].reshape(smoothed.shape[0],1), [1, smoothingWindowSize])
            # replace nans with large values (so they don't get picked up as mins)
            smoothed[np.isnan(smoothed)] = np.max(smoothed)*2
            baseline = ndimage.grey_erosion(smoothed, footprint=np.ones((1, smoothingWindowSize)))
            plt.figure,plt.plot(baseline[20,:]),plt.plot(smoothed[20,:]), plt.show

            # # calculate dF/F
            dF = (F-baseline) / baseline
            # get times of each frame
            depthFrameTimes = frameTimes[iDepth+1:depthCount:len(frameTimes)]
            depthFrameTimes = depthFrameTimes[:dF.shape[1]]
            # resample to get desired sampling rate
            dF = interp1d(depthFrameTimes, dF.T, axis=0, fill_value="extrapolate")(outputTimes).T
            F = interp1d(depthFrameTimes, F.T, axis=0, fill_value="extrapolate")(outputTimes).T
            Spks = interp1d(depthFrameTimes, Spks.T, axis=0, fill_value="extrapolate")(outputTimes).T

            if dF.shape[1] == 1:
                dF = dF.T

            # pick out valid cells
            alldF[iCh] = np.concatenate([alldF[iCh], dF])
            allF[iCh] = np.concatenate([allF[iCh], F])
            allSpikes[iCh] = np.concatenate([allSpikes[iCh], Spks])

            allDepths[iCh] = np.concatenate([allDepths[iCh], np.tile(iDepth, (np.sum(cellValid[:, 0]), 1))])
            allRoiPix[iCh][iDepth+1] = roiPix
            allRoiMaps[iCh][iDepth+1] = roiMap

            allFOV[iCh] = Fall['ops']['meanImg']

    print('Saving 2-photon data...')

    # save as CSV
    for iCh in range(len(alldF)):
        np.savetxt(os.path.join(recordingsRoot, f'dF_{iCh}.csv'), np.transpose([outputTimes, alldF[iCh]]), delimiter=',')
        np.savetxt(os.path.join(recordingsRoot, f'F_{iCh}.csv'), np.transpose([outputTimes, allF[iCh]]), delimiter=',')
        np.savetxt(os.path.join(recordingsRoot, f'Spikes_{iCh}.csv'), np.transpose([outputTimes, allSpikes[iCh]]), delimiter=',')
        np.savetxt(os.path.join(recordingsRoot, f'roi_{iCh}.csv'), allRoiMaps[iCh], delimiter=',')
        np.savetxt(os.path.join(recordingsRoot, f'fov_{iCh}.csv'), allFOV[iCh], delimiter=',')

    # save for MATLAB
    s2pData = {'alldF': alldF, 'allF': allF, 'allDepths': allDepths, 'allRoiPix': allRoiPix,
            'allRoiMaps': allRoiMaps, 'meanFrame': Fall['ops']['meanImg'], 't': outputTimes}
    sio.savemat(os.path.join(recordingsRoot, 's2pData.mat'), s2pData)
