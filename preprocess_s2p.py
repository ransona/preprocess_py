import os
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
import dcnv
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import HuberRegressor
from scipy.ndimage import minimum_filter1d, maximum_filter1d
from scipy.ndimage import percentile_filter
from numpy.lib.stride_tricks import sliding_window_view
from scipy import interpolate
from scipy import signal
from scipy.io import loadmat
import matplotlib.pyplot as plt
import pickle
import organise_paths
from tqdm import tqdm
from joblib import Parallel, delayed
# OASIS imports
import warnings
warnings.filterwarnings("ignore", module="oasis")
from oasis.functions import deconvolve
from oasis.functions import gen_data, gen_sinusoidal_data, estimate_parameters
from oasis.plotting import simpleaxis
from oasis.oasis_methods import oasisAR1, oasisAR2
import gc

import os

# info:
# spike deconvolution:
# this is performed when suite2p runs but it produces a spike vector which is proportional to the F signal, not dF/F.
# to remedy this, here we calculate dF/F and run spike deconvolution on dF/F traces. note that values < 0 in dF/F
# trace will produce zero spike rate by algorithm, and therefore baselining such that estimated resting state has 
# mean = 0 means that a large part of the trace during baseline can not produce spikes. this seems to be how the 

def run_preprocess_s2p(userID, expID, neuropil_coeff_config = np.nan):
    animalID, remote_repository_root, \
    processed_root, exp_dir_processed, \
        exp_dir_raw = organise_paths.find_paths(userID, expID)
    exp_dir_processed_recordings = os.path.join(processed_root, animalID, expID,'recordings')

    # load the pipeline command configuration file
    try:
        with open(os.path.join(exp_dir_processed,'step2_config.pickle'), "rb") as file: 
            step2_config = pickle.load(file)
    except:
        step2_config = {}

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

    # Neuropil weight used is in this priority order:
    # 1. provided as argument to function
    # 2. provided in config file
    # 3. default to 0.7
    # get the neuropil coefficient from config if present or default to 0.7
    if neuropil_coeff_config is not np.nan:
        # provided as argument to function 
        neuropilWeight = neuropil_coeff_config
        print(f"Using neuropil weight provided as argument to function: {neuropilWeight}")
    else:
        # attempt to get from config file
        neuropil_coeff_config = step2_config.get('settings', {}).get('neuropil_coeff')
        # Initialize neuropilWeight with a default two-element list
        if neuropil_coeff_config is None:
            neuropilWeight = [0.7, 0.7]  # Default values if neuropil_coeff_config is not provided
            print(f"Using default neuropil weight : {neuropilWeight}")
        elif isinstance(neuropil_coeff_config, float):
            neuropilWeight = [neuropil_coeff_config, neuropil_coeff_config]  # Duplicate float if single value
            print(f"Using neuropil weight from config file: {neuropilWeight}")
        elif isinstance(neuropil_coeff_config, (list, tuple)):
            if len(neuropil_coeff_config) == 1:
                neuropilWeight = [neuropil_coeff_config[0], neuropil_coeff_config[0]]  # Duplicate single element
            else:
                neuropilWeight = list(neuropil_coeff_config[:2])  # Take the first two elements if it's longer
            print(f"Using neuropil weight from config file: {neuropilWeight}")
        else:
            raise TypeError("Unexpected type for neuropil_coeff_config. Expected float, list, or tuple.")

    # initiate these as dict:
    alldF = {}
    allBaseline = {}
    allF = {}
    allSpikes = {}          # S2P spike output
    #all_oasis_spikes = {}   # OASIS spike output (computed below)
    #all_oasis_dF = {}       # OASIS inferred calcium (computed below)
    # tokenised data
    #all_tokenised_oasis_dF = {}
    #all_tokenised_oasis_spikes = {}
    all_tokenised_dF_spikes = {}
    all_tokenised_dF = {}   

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
    # frameTimes will be the mid-frame time
    frameTimes = np.squeeze(tl_time)[np.where(np.diff(neuralFramesPulses)==1)[0]]
    # frame_start_times will be the time each frame trigger happens
    frame_start_times = frameTimes.copy()
    # convert frame start times to midframe times
    time_diffs = time_diffs = np.append(np.diff(frameTimes), np.diff(frameTimes)[-1])
    frameTimes = frameTimes + (time_diffs/2)

    framePulsesPerDepth = len(frameTimes)/depthCount
    frameRate = 1/np.median(np.diff(frameTimes))
    frameRatePerPlane = frameRate/depthCount
    frame_duration = np.median(time_diffs)
    
    # determine timeline times when we want the Ca signal of each cell
    # +1 and -1 are because we want to make sure we only include frame
    # times which are available at all depths
    outputTimes = np.arange(frameTimes[0]+1, frameTimes[-1]-1, 1/resampleFreq)
    print()
    for iCh in range(len(dataPath)):
        accumulated_rois = 0
        for iDepth in range(depthCount):
            print('Starting Ch'+str(iCh)+' Depth '+str(iDepth))
            allRoiPix[iCh] = {}
            #allRoiPix[iCh][iDepth] = np.array([])
            allRoiMaps[iCh] = {}
            #allRoiMaps[iCh][iDepth] = np.array([])
            allFOV[iCh] = {}
            # load s2p data
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
                # need to ensure these are 2d even if only one cell
                Fneu_valid = np.squeeze(Fneu[np.where(cellValid==1), :])
                F_valid = np.squeeze(Fall[np.where(cellValid==1), :])
                Spks_valid = np.squeeze(spks[np.where(cellValid==1), :])
                Fneu_valid = Fneu_valid[np.newaxis,:]
                F_valid = F_valid[np.newaxis,:]
                Spks_valid = Spks_valid[np.newaxis,:]

            xpix, ypix = [], []
            validCellIDs = np.where(cellValid==1)[0]

            # now get the x and ypix of each valid cell 
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

            # offset whole video to ensure all F and neuropil values are at least 20.
            FMins = np.min(F_valid, axis=1)
            FMins_neuropil = np.min(Fneu_valid, axis=1)
            min_all_rois = np.min([np.min(FMins), np.min(FMins_neuropil)])          
            if min_all_rois < 20:
                offset_value = min_all_rois * -1 + 20
                F_valid = F_valid + offset_value
                Fneu_valid = Fneu_valid + offset_value
                print('Offsetting all F and neuropil by', offset_value, 'to ensure min > 20')

            print('Subtracting neuropil with weight', neuropilWeight[iCh])
            F_valid = F_valid - (Fneu_valid * neuropilWeight[iCh])
            print('Done neuropil subtraction.')

            # Check again the F values after neuropil subtraction are >=20
            FMins = np.min(F_valid, axis=1)

            if np.min(FMins) < 20:
                print('Frame mean and neuropil subtraction give ROIs with F < 20')
                print('Offsetting all F by', (np.min(FMins)*-1)+20)
                F_valid = F_valid + (np.min(FMins)*-1)+20
                
                
            # FMins = np.min(F_valid, axis=1)
            # plt.subplot(1, 2, 2)
            # plt.hist(FMins)
            # plt.title(['Distribution of F values', 'of ROIS after forcing > 20'])
            # plt.draw()

            # do merge was removed from here - this was bill's code for merging correlated rois
            # what to do if not merging
            # make a roi map for the depth that can be used for longitudinal imaging etc
            print("Creating ROI map...")
            roiPix = []
            # make a blank roi map
            roiMap = np.zeros(np.shape(s2p_ops['meanImg']))

            for iRoi in range(F_valid.shape[0]):
                # collect pix in ROI
                roiPix.append(np.ravel_multi_index((ypix[iRoi],xpix[iRoi]), np.shape(s2p_ops['meanImg'])))
                # label ROI map
                roiMap[ypix[iRoi], xpix[iRoi]] = iRoi+1

            print("ROI map created.")
            # dF/F calculation
            print("Calculating dF/F baseline...")
            # this window is for smoothing
            smoothingWindowSize = (1 * frameRatePerPlane).round().astype(int)
            # this window if for percentile should equate to 30 secs
            baseline_min_window_size = (30 * frameRatePerPlane).round().astype(int)
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
            # baseline = np.apply_along_axis(lambda smoothed: np.minimum.accumulate(np.concatenate([smoothed, np.repeat(smoothed[-1], baseline_min_window_size)]))[baseline_min_window_size:], axis=1, arr=smoothed)
            # baseline = minimum_filter1d(smoothed, size=baseline_min_window_size, axis=1)
            # baseline = np.apply_along_axis(lambda smoothed: np.pad(np.percentile(sliding_window_view(smoothed, baseline_min_window_size), 20, axis=1), (baseline_min_window_size - 1, 0), mode='edge'), axis=1, arr=smoothed)
            # print("dF/F baseline calculated.")

            def efficient_causal_sliding_percentile_2d(data, window, step=1, percentile=20):
                """
                Applies causal sliding percentile baseline correction across time axis for each neuron.
                Input:
                    data: np.ndarray of shape (neurons, time)
                    window: window size (in samples)
                    step: stride for percentile calculation
                    percentile: which percentile to compute
                Output:
                    baseline: np.ndarray of shape (neurons, time)
                """
                neurons, T = data.shape
                idx = np.arange(window - 1, T, step)
                start_idx = idx - window + 1
                valid = start_idx >= 0
                idx = idx[valid]
                start_idx = start_idx[valid]

                # Preallocate output
                baseline = np.zeros_like(data)

                for n in tqdm(range(neurons), desc="Computing baselines"):
                    # Extract sliding windows
                    windows = np.array([data[n, i:j + 1] for i, j in zip(start_idx, idx)])
                    p_values = np.percentile(windows, percentile, axis=1)

                    # Interpolate to full length
                    full_idx = np.arange(T)
                    interp = np.interp(full_idx, idx, p_values)
                    baseline[n] = interp

                return baseline
        
            # make step size 30 seconds   
            pencentile_window_step_size_secs = 10
            pencentile_window_step_size_samples = (pencentile_window_step_size_secs * frameRatePerPlane).round().astype(int) 
            f_percentile = 10   
            baseline = efficient_causal_sliding_percentile_2d(data = smoothed, window = baseline_min_window_size, step=pencentile_window_step_size_samples, percentile=f_percentile) 
            print("dF/F baseline2 calculated.")

            # debugging plots for baseline calculation
            # n_cells_to_show = 10
            # cell_idx = np.linspace(0,baseline.shape[0]*0.1,n_cells_to_show,dtype=int)
            
            # fig, axes = plt.subplots(n_cells_to_show,1)
            # for k,iCell in enumerate(cell_idx):
            #     axes[k].plot(F_valid[iCell,0:3000],color="red")
            #     axes[k].plot(Spks_valid[iCell,0:3000]+baseline[iCell,0:3000],color="black")
            #     axes[k].plot(baseline[iCell,0:3000],color="blue")
                
            plt.show()
                
             # # calculate dF/F
            print("Calculating dF/F...")           
            dF = (F_valid-baseline) / baseline
            dF_spikes = Spks_valid / baseline
            print("dF/F calculated.")

            # debugging plots for dF/F calculation
            # n_cells_to_show = 10
            # cell_idx = np.linspace(0,baseline.shape[0]*0.1,n_cells_to_show,dtype=int)
            
            # fig, axes = plt.subplots(n_cells_to_show,1)
            # for k,iCell in enumerate(cell_idx):
            #     axes[k].plot(dF[iCell,0:3000],color="red")
            #     axes[k].plot(dF_spikes[iCell,0:3000],color="black")
                
            # plt.show()            

            # preallocate to store spike deconvolution output
            oasis_spikes = np.zeros(dF.shape)
            oasis_calcium = np.zeros(dF.shape)

            # OASIS deconvolution can be truned back on with this block of code
            # ///////////// OASIS START ///////////// 

            # def process_cell(iCell,cell_trace):
            #     y_offset = np.min(cell_trace)
            #     cell_trace = cell_trace - y_offset
            #     with warnings.catch_warnings():
            #         warnings.simplefilter("ignore")
            #         c, s, b, g, lam = deconvolve(cell_trace, penalty=0)
            #     # s[s < 0.05] = 0
            #     # add offset and baseline back on to calcium signal estimation of model
            #     c = c + b + y_offset
            #     return iCell, s, c

            # # Parallel execution
            # if os.getenv("DEBUGPY_RUNNING"):
            #     # single-threaded when debugging
            #     print("Debugging mode detected, running single-threaded.")
            #     for iCell in tqdm(range(dF.shape[0]), desc="Processing cells", unit="cell"):
            #         # add a percentage complete display that shows progress bar
            #         pass
            #         # if np.mod(iCell, 10) == 0:
            #         #     print(f'Cell {iCell} of {dF.shape[0]} ({(iCell/dF.shape[0])*100:.1f}%)')
            #         # pull out current cell
            #         cell_trace = dF[iCell,:]# dF[iCell,:]
            #         # add an offset to ensure all values are positive
            #         y_offset = np.min(cell_trace)
            #         cell_trace = cell_trace - y_offset
            #         # deconvolve
            #         c, s, b, g, lam = deconvolve(cell_trace, penalty=0)
            #         # threshold under assumption that transitions of < 0.05 are noise based on approx
            #         # min 1AP response described here https://pubmed.ncbi.nlm.nih.gov/36922596/#&gid=article-figures&pid=fig-4-uid-3
            #         # s[s<0.05] = 0 # removed 02.12.25 to see if we were losing information
            #         # add offset back on to calcium signal estimation of model
            #         c = c + b + y_offset
            #         # store data
            #         oasis_spikes[iCell,:] = s
            #         oasis_calcium[iCell,:] = c
            # else:
            #     print("Running multi-threaded.")
            #     results = Parallel(n_jobs=-1, prefer="processes")(
            #         delayed(process_cell)(iCell,dF[iCell,:])
            #         for iCell in tqdm(range(dF.shape[0]), desc="Processing cells", unit="cell")
            #     )
            #     # Collect results
            #     for iCell, s, c in results:
            #         oasis_spikes[iCell, :] = s
            #         oasis_calcium[iCell, :] = c  

            # print("Devonvolution complete.")

            # ///////////// OASIS END ///////////// 

            

            # debug plots for deconvolution
            # n_cells_to_show = 5
            # cell_idx = np.linspace(0,baseline.shape[0]*0.1,n_cells_to_show,dtype=int)            
            # fig, axes = plt.subplots(n_cells_to_show,1)
            # for k,iCell in enumerate(cell_idx):
            #     axes[k].plot(dF[iCell,0:3000],color="red")
            #     axes[k].plot(dF_spikes[iCell,0:3000],color="black")
            #     axes[k].plot(oasis_spikes[iCell,0:3000],color="blue")
            # axes[k].legend(['dF/F','s2p spikes','oasis spikes'])
            # plt.show()

            

            ####################
            # get times of each frame
            # mid frame time
            depthFrameTimes = frameTimes[iDepth:len(frameTimes):depthCount]
            # start frame time of current and next frame
            depth_frame_start_times = frame_start_times[iDepth:len(frame_start_times):depthCount]
            next_depth_frame_times = depth_frame_start_times + frame_duration
            
            # make sure there are not more times than frames or vice versa
            min_frame_count = min(dF.shape[1],len(depthFrameTimes))
            if dF.shape[1]<len(depthFrameTimes):
                print('Warning: less frames in tif than frame triggers, diff = ' + str(len(depthFrameTimes)-dF.shape[1]))
            elif dF.shape[1]>len(depthFrameTimes):
                print('Warning: less frame triggers than frames in tif, diff = ' + str(dF.shape[1]-len(depthFrameTimes)))

            depthFrameTimes = depthFrameTimes[:min_frame_count]
            depth_frame_start_times = depth_frame_start_times[:min_frame_count]
            next_depth_frame_times = next_depth_frame_times[:min_frame_count]

            dF = dF[:,:min_frame_count]
            dF_spikes = dF_spikes[:,:min_frame_count]
            F_valid = F_valid[:,:min_frame_count]
            oasis_spikes = oasis_spikes[:,:min_frame_count]
            oasis_calcium = oasis_calcium[:,:min_frame_count]

            # resample to get desired sampling rate. note '.T' is the transpose to get matrix orientation correct for interpolation
            print("Resampling to desired output frequency...")
            depthFrameTimes = np.asarray(depthFrameTimes)
            outputTimes = np.asarray(outputTimes)
            # finds, for each outputTimes entry, the index of the latest depthFrameTimes value that is ≤ that output time
            idx = np.searchsorted(depthFrameTimes, outputTimes, side="right") - 1
            idx = np.clip(idx, 0, len(depthFrameTimes) - 1)  

            dF_resampled = dF[:, idx]
            F_resampled = F_valid[:, idx]
            Spks_resampled = dF_spikes[:, idx]
            Baseline_resampled = baseline[:, idx]
            oasis_spikes_resampled = oasis_spikes[:, idx]
            oasis_calcium_resampled = oasis_calcium[:, idx]

            print("Resampling complete.")
            # if there is only one cell ensure it is rotated to (cell,time) orientation
            if len(dF_resampled.shape) == 1:
                # add the new axis to make sure it is (cell,time)
                dF_resampled = dF_resampled[np.newaxis,:]
                F_resampled = F_resampled[np.newaxis,:]
                Spks_resampled = Spks_resampled[np.newaxis,:]
                Baseline_resampled = Baseline_resampled[np.newaxis,:]
                oasis_spikes_resampled = oasis_spikes_resampled[np.newaxis,:]
                oasis_calcium_resampled = oasis_calcium_resampled[np.newaxis,:]

            # pick out valid cells
            allRoiPix[iCh] = {}

            # construct tokenised neural response representation where you have cell ID,timestamp,spike/df value
            # Inputs:
            # y_coords: shape (n_cells,)
            # frame_height: scalar
            # activity: shape (n_cells, n_timepoints)
            # depthFrameTimes: shape (n_timepoints,)
            # next_depth_frame_times: shape (n_timepoints,)
            print("Constructing tokenised neural activity matrix.")
            n_cells, n_timepoints = dF_spikes.shape

            # Scale y to fraction (0 to 1)
            frame_height = np.shape(s2p_ops['meanImg'])[0]
            # for each neuron calculate the median y coordinate (distance from top of frame)
            y_coords = np.array([np.median(ypix[i]) for i in range(len(ypix))]) # shape: (n_cells,)
            fraction = y_coords / frame_height  # shape: (n_cells,)
            # Compute difference between frame times
            time_deltas = next_depth_frame_times - depth_frame_start_times  # shape: (n_timepoints,)

            # Compute sample times:
            # sample_time = depthFrameTimes + fraction * (next - depth)
            sample_times = depth_frame_start_times[None, :] + fraction[:, None] * time_deltas[None, :]  # shape: (n_cells, n_timepoints)

            # Flatten (all first row first, etc)
            cell_ids = np.repeat(np.arange(n_cells), n_timepoints)
            cell_ids = cell_ids + accumulated_rois
            accumulated_rois = accumulated_rois + n_cells
            sample_times_flat = sample_times.flatten()
            # activity from suite2p spike deconvolution
            activity_flat_dF_spikes = dF_spikes.flatten()
            # activity from raw dF/F
            activity_flat_dF = dF.flatten()

            # Combine into final tokenized matrices [cell_id, sample_time, activity] of depth
            tokenised_dF_spikes = np.stack((cell_ids, sample_times_flat, activity_flat_dF_spikes), axis=1)
            tokenised_dF = np.stack((cell_ids, sample_times_flat, activity_flat_dF), axis=1)
            
            print("Tokenised neural activity matrix constructed.")

            # accumulate data across depths
            print("Accumulating data across depths...")
            if dF_resampled.shape[0] > 0:
                if iCh not in alldF:
                    alldF[iCh] = [dF_resampled]
                    allF[iCh] = [F_resampled]
                    allBaseline[iCh] = [Baseline_resampled]
                    allSpikes[iCh] = [Spks_resampled]
                    allDepths[iCh] = [np.tile(iDepth, (np.sum(cellValid[:]).astype(int), 1))]
                    #all_oasis_spikes[iCh] = [oasis_spikes_resampled]
                    #all_oasis_dF[iCh] = [oasis_calcium_resampled]
                    #all_tokenised_oasis_dF[iCh] = [tokenised_oasis_dF]
                    all_tokenised_dF_spikes[iCh] = [tokenised_dF_spikes]
                    dF_spikes
                    all_tokenised_dF[iCh] = [tokenised_dF]
                else:
                    alldF[iCh].append(dF_resampled)
                    allF[iCh].append(F_resampled)
                    allBaseline[iCh].append(Baseline_resampled)
                    allSpikes[iCh].append(Spks_resampled)
                    allDepths[iCh].append(np.tile(iDepth, (np.sum(cellValid[:]).astype(int), 1)))
                    #all_oasis_spikes[iCh].append(oasis_spikes_resampled)
                    #all_oasis_dF[iCh].append(oasis_calcium_resampled)
                    #all_tokenised_oasis_dF[iCh].append(tokenised_oasis_dF)
                    #all_tokenised_oasis_spikes[iCh].append(tokenised_oasis_spikes)
                    all_tokenised_dF_spikes[iCh].append(tokenised_dF_spikes)
                    all_tokenised_dF[iCh].append(tokenised_dF)

                    del dF_resampled, F_resampled, Baseline_resampled, Spks_resampled
                    del oasis_spikes_resampled, oasis_calcium_resampled
                    del tokenised_oasis_dF, tokenised_oasis_spikes, tokenised_dF
                    del dF,dF_spikes,F_valid,Spks_valid,baseline
                    del oasis_spikes, oasis_calcium
                    gc.collect()

            print("Accumulation complete.")
            allRoiPix[iCh][iDepth] = roiPix
            allRoiMaps[iCh][iDepth] = roiMap
            allFOV[iCh][iDepth] = s2p_ops['meanImg']

    print('Concatinating final output...')
    for iCh in alldF:
        alldF[iCh] = np.concatenate(alldF[iCh], axis=0)
        allF[iCh] = np.concatenate(allF[iCh], axis=0)
        allBaseline[iCh] = np.concatenate(allBaseline[iCh], axis=0)
        allSpikes[iCh] = np.concatenate(allSpikes[iCh], axis=0)
        allDepths[iCh] = np.concatenate(allDepths[iCh], axis=0)
        all_oasis_spikes[iCh] = np.concatenate(all_oasis_spikes[iCh], axis=0)
        all_oasis_dF[iCh] = np.concatenate(all_oasis_dF[iCh], axis=0)
        all_tokenised_oasis_dF[iCh] = np.concatenate(all_tokenised_oasis_dF[iCh], axis=0)
        all_tokenised_oasis_spikes[iCh] = np.concatenate(all_tokenised_oasis_spikes[iCh], axis=0)
        all_tokenised_dF[iCh] = np.concatenate(all_tokenised_dF[iCh], axis=0)

    print('Saving 2-photon data...')
    # save
    for iCh in range(len(alldF)):
        # make a dict where all of the experiment data is stored
        ca_data = {}
        ca_data_tokenised = {}
        ca_data_oasis= {}
        ca_data['dF']           = alldF[iCh].astype(np.float32)
        ca_data['F']            = allF[iCh].astype(np.int16)
        ca_data['Spikes']       = allSpikes[iCh].astype(np.float32)
        ca_data['Baseline']     = allBaseline[iCh].astype(np.int16)
        # oasis outputs
        ca_data_oasis['oasis_dF']     = (all_oasis_dF[iCh]*100).astype(np.int16)
        ca_data_oasis['oasis_spikes'] = (all_oasis_spikes[iCh]*100).astype(np.uint16)


        # tokenised data
        # sort tokenised matrices by time.
        # CRITICAL NOTE: sorting by token time alone does not give consistent cell order due to rounding errors
        #                of neuron timestamps that are very close in time. This can cause swapping of neuron order!
        all_tokenised_oasis_spikes[iCh] = all_tokenised_oasis_spikes[iCh][np.lexsort((all_tokenised_oasis_spikes[iCh][:, 0], np.round(all_tokenised_oasis_spikes[iCh][:, 1], 9)))]
        ca_data_tokenised['tokenised_oasis_spikes'] = all_tokenised_oasis_spikes[iCh].astype(np.float32)
        
        # other data
        ca_data['Depths']       = allDepths[iCh]
        ca_data['AllRoiPix']    = allRoiPix[iCh]
        ca_data['AllRoiMaps']   = allRoiMaps[iCh]
        ca_data['AllFOV']       = allFOV[iCh]
        ca_data['t']            = outputTimes

        ca_data_oasis['Depths']       = allDepths[iCh]
        ca_data_oasis['AllRoiPix']    = allRoiPix[iCh]
        ca_data_oasis['AllRoiMaps']   = allRoiMaps[iCh]
        ca_data_oasis['AllFOV']       = allFOV[iCh]
        ca_data_oasis['t']            = outputTimes        
        # write out main data
        output_filename = 's2p_ch' + str(iCh)+'.pickle'
        pickle_out = open(os.path.join(exp_dir_processed_recordings,output_filename),"wb")
        pickle.dump(ca_data, pickle_out)
        pickle_out.close()
        # write out tokenised data
        output_filename = 's2p_tokenised_ch' + str(iCh)+'.pickle'
        pickle_out = open(os.path.join(exp_dir_processed_recordings,output_filename),"wb")
        pickle.dump(ca_data_tokenised, pickle_out)
        pickle_out.close()
        # write out oasis data
        output_filename = 's2p_oasis_ch' + str(iCh)+'.pickle'
        pickle_out = open(os.path.join(exp_dir_processed_recordings,output_filename),"wb")
        pickle.dump(ca_data_oasis, pickle_out)
        pickle_out.close()

    print('2-photon preprocessing done')

# for debugging:
def main():
    # debug mode
    # allExpIDs = [
    # '2025-03-12_01_ESPM126', 
    # '2025-03-13_02_ESPM126',
    # '2025-03-26_01_ESPM126', '2025-03-26_02_ESPM126',
    # '2025-04-01_01_ESPM127', 
    '2025-04-01_02_ESPM127',
    # '2025-06-12_02_ESPM135', '2025-06-12_04_ESPM135',
    # '2025-06-13_01_ESPM135', '2025-06-13_02_ESPM135',
    # '2025-07-02_03_ESPM135', '2025-07-02_05_ESPM135',
    # '2025-07-08_04_ESPM152', '2025-07-08_05_ESPM152',
    # '2025-07-04_04_ESPM154', '2025-07-04_06_ESPM154',
    # '2025-07-07_05_ESPM154', '2025-07-07_06_ESPM154',
    # '2025-07-11_02_ESPM154', '2025-07-11_03_ESPM154',
    # # rivalry
    # '2025-07-17_01_ESPM154', '2025-07-17_04_ESPM154',
    # '2025-08-07_01_ESPM163', '2025-08-07_05_ESPM163',
    # '2025-08-12_01_ESPM164', '2025-08-12_04_ESPM164'
    # ]
    # userID = 'pmateosaparicio'

    # for expID in allExpIDs:    
    #     run_preprocess_s2p(userID, expID, neuropil_coeff_config=[0.7 , 0.7]) 

    allExpIDs = [
    # rivalry
    '2025-07-17_01_ESPM154', '2025-07-17_04_ESPM154',
    '2025-08-07_01_ESPM163', '2025-08-07_05_ESPM163',
    '2025-08-12_01_ESPM164', '2025-08-12_04_ESPM164'
    ]
    userID = 'melinatimplalexi'

    for expID in allExpIDs:    
        run_preprocess_s2p(userID, expID, neuropil_coeff_config=[0.7 , 0.7])         

    allExpIDs = [
    '2025-02-26_02_ESPM126']
    userID = 'adamranson'    

    for expID in allExpIDs:    
        run_preprocess_s2p(userID, expID, neuropil_coeff_config=[0.7 , 0.7]) 


if __name__ == "__main__":
    main()