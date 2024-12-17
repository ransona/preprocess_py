import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy import interpolate
from scipy import signal
from scipy.io import loadmat
import matplotlib.pyplot as plt
import pickle
import organise_paths

def run_preprocess_s2p(userID, expID):
    animalID, remote_repository_root, \
    processed_root, exp_dir_processed, \
        exp_dir_raw = organise_paths.find_paths(userID, expID)
    exp_dir_processed_recordings = os.path.join(processed_root, animalID, expID,'recordings')

    # load the pipeline command configuration file
    with open(os.path.join(exp_dir_processed,'step2_config.pickle'), "rb") as file: 
        step2_config = pickle.load(file)


    # load timeline
    Timeline = loadmat(os.path.join(exp_dir_raw, expID + '_Timeline.mat'))
    Timeline = Timeline['timelineSession']
    # get timeline file in a usable format after importing to python
    tl_chNames = Timeline['chNames'][0][0][0][0:]
    tl_daqData = Timeline['daqData'][0,0]
    tl_time    = Timeline['time'][0][0]

    resampleFreq = 30

    # subtract overall frame or not
    subtract_overall_frame_config = step2_config.get('settings', {}).get('subtract_overall_frame')
    if subtract_overall_frame_config is None:
        subtract_overall_frame = False
    else:
        subtract_overall_frame = subtract_overall_frame_config

    # get the neuropil coefficient from config if present or default to 0.7
    neuropil_coeff_config = step2_config.get('settings', {}).get('neuropil_coeff')
    # Initialize neuropilWeight with a default two-element list
    if neuropil_coeff_config is None:
        neuropilWeight = [0.7, 0.7]  # Default values if neuropil_coeff_config is not provided
    elif isinstance(neuropil_coeff_config, float):
        neuropilWeight = [neuropil_coeff_config, neuropil_coeff_config]  # Duplicate float if single value
    elif isinstance(neuropil_coeff_config, (list, tuple)):
        if len(neuropil_coeff_config) == 1:
            neuropilWeight = [neuropil_coeff_config[0], neuropil_coeff_config[0]]  # Duplicate single element
        else:
            neuropilWeight = list(neuropil_coeff_config[:2])  # Take the first two elements if it's longer
    else:
        raise TypeError("Unexpected type for neuropil_coeff_config. Expected float, list, or tuple.")

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
    print()
    for iCh in range(len(dataPath)):

        for iDepth in range(depthCount):
            print('Starting Ch'+str(iCh)+' Depth '+str(iDepth))
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
            if sum(cellValid)>1:
                Fneu_valid = np.squeeze(Fneu[np.where(cellValid==1), :])
                F_valid = np.squeeze(Fall[np.where(cellValid==1), :])
                Spks_valid = np.squeeze(spks[np.where(cellValid==1), :])
            else:
                Fneu_valid = np.squeeze(Fneu[np.where(cellValid==1), :])
                F_valid = np.squeeze(Fall[np.where(cellValid==1), :])
                Spks_valid = np.squeeze(spks[np.where(cellValid==1), :])
                Fneu_valid = Fneu_valid[np.newaxis,:]
                F_valid = F_valid[np.newaxis,:]
                Spks_valid = Spks_valid[np.newaxis,:]

            xpix, ypix = [], []
            validCellIDs = np.where(cellValid==1)[0]
            # this is to later store the x and ypix of each valid cell
            for iCell in range(len(validCellIDs)):
                currentCell = validCellIDs[iCell]
                xpix.append(s2p_stat[currentCell]['xpix'])
                ypix.append(s2p_stat[currentCell]['ypix'])

            # remove potential stimulus artifact - i.e. mean of frame which
            # is extracted above
            if subtract_overall_frame:
                Fneu_valid = Fneu_valid - np.tile(meanFrameTimecourse, (Fneu_valid.shape[0], 1))
                F_valid = F_valid - np.tile(meanFrameTimecourse, (F_valid.shape[0], 1))

            # neuropil subtraction
            F_valid = F_valid - (Fneu_valid * neuropilWeight[iCh])

            # Ensure min(corrected F) > 10
            FMins = np.min(F_valid, axis=1)
            # plt.subplot(1, 2, 1)
            # plt.hist(FMins)
            # plt.title(['Distribution of original', 'F values of ROIS'])

            if np.min(FMins) < 20:
                print('Frame mean and neuropil subtraction give ROIs with F < 20')
                print('Offsetting all F by', (np.min(FMins)*-1)+20)
                F_valid = F_valid + (np.min(FMins)*-1)+20
                
                
            FMins = np.min(F_valid, axis=1)
            # plt.subplot(1, 2, 2)
            # plt.hist(FMins)
            # plt.title(['Distribution of F values', 'of ROIS after forcing > 20'])
            # plt.draw()

            # do merge was removed from here - this was bill's code for merging correlated rois
            # what to do if not merging
            # make a roi map for the depth that can be used for longitudinal imaging etc
            roiPix = []
            # make a blank roi map
            roiMap = np.zeros(np.shape(s2p_ops['meanImg']))

            for iRoi in range(F_valid.shape[0]):
                # collect pix in ROI
                roiPix.append(np.ravel_multi_index((ypix[iRoi],xpix[iRoi]), np.shape(s2p_ops['meanImg'])))
                # label ROI map
                roiMap[ypix[iRoi], xpix[iRoi]] = iRoi+1
            # plt.figure
            # plt.imshow(roiMap), plt.show()
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
            # make sure there are not more times than frames or vice versa
            min_frame_count = min(dF.shape[1],len(depthFrameTimes))
            if dF.shape[1]<len(depthFrameTimes):
                print('Warning: less frames in tif than frame triggers, diff = ' + str(len(depthFrameTimes)-dF.shape[1]))
            elif dF.shape[1]>len(depthFrameTimes):
                print('Warning: less frame triggers than frames in tif, diff = ' + str(dF.shape[1]-len(depthFrameTimes)))
        
            depthFrameTimes = depthFrameTimes[:min_frame_count]
            dF = dF[:,:min_frame_count]
            F_valid = F_valid[:,:min_frame_count]
            Spks_valid = Spks_valid[:,:min_frame_count]
            
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

    # save
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
        ca_data['t']            = outputTimes
        output_filename = 's2p_ch' + str(iCh)+'.pickle'
        pickle_out = open(os.path.join(exp_dir_processed_recordings,output_filename),"wb")
        pickle.dump(ca_data, pickle_out)
        pickle_out.close()

    print('2-photon preprocessing done')