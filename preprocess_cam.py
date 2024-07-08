import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.io import loadmat
import organise_paths
import pickle

def preprocess_cam_run(userID, expID):
    print('Starting preprocess_cam_run...')
    animalID, remote_repository_root, \
    processed_root, exp_dir_processed, \
        exp_dir_raw = organise_paths.find_paths(userID, expID)
    exp_dir_processed_recordings = os.path.join(processed_root, animalID, expID,'recordings')

    # load timeline
    Timeline = loadmat(os.path.join(exp_dir_raw, expID + '_Timeline.mat'))
    Timeline = Timeline['timelineSession']
    # get timeline file in a usable format after importing to python
    tl_chNames = Timeline['chNames'][0][0][0][0:]
    tl_daqData = Timeline['daqData'][0,0]
    tl_time    = Timeline['time'][0][0][0]

    # Here load meta dta acquired with video frames. This contains frame times.
    # This is either in a .mat file (pre 08/07/24) or in a .pickle files (after this date)
    if os.path.isfile(os.path.join(exp_dir_raw, (expID + '_eyeMeta1.mat'))):
        # Load the MAT file
        mat_contents = loadmat(os.path.join(exp_dir_raw, (expID + '_eyeMeta1.mat')))
        eTrackData = mat_contents['eTrackData']
        eye_frameTimes = eTrackData['frameTimes'][0][0][0] # in one row array
        # eye_frameCount = eTrackData['frameCount'][0][0][0][0] # single number
    else:
        # Load the pickle
        pickle_contents = pickle.load(open(os.path.join(exp_dir_raw, (expID + '_eyeMeta1.pickle')), "rb"))
        eye_frameTimes = np.array(pickle_contents['frame_times'])

    # Find the index of the 'EyeCamera' channel in the Timeline data
    camIdx = np.where(np.isin(tl_chNames, 'EyeCamera'))[0][0]
    # Extract the camera pulse trace and the corresponding frame pulse times
    camPulseTrace = (tl_daqData[:, camIdx] > 2.5).astype(int)
    framePulseTimes = tl_time[np.where(np.diff(camPulseTrace) == 1)] # each pos going edge = 200 frames (except first)

    # Do a quality check on the frame pulse times
    if np.min(np.diff(framePulseTimes)) < 14:
        plt.figure()
        if len(tl_time) > 100000:
            plt.plot(tl_time[:100000], tl_daqData[:100000, camIdx])
        else:
            plt.plot(tl_time[0:], tl_daqData[0:, camIdx])
        plt.title('Eye camera timing pulses (ch' + str(camIdx) + ' of DAQ)')
        plt.xlabel('Time (secs)')
        plt.ylabel('Voltage (volts)')
        print('The timing pulses on the eye camera look faulty - see the figure')
        raise Exception('The timing pulses on the eye camera look faulty - see the figure')

    # Shift the logged frame times to align with the frame pulse times
    loggedFrameTimes = eye_frameTimes - eye_frameTimes[0]
    loggedFrameTimes = loggedFrameTimes + framePulseTimes[0]

    # Periodically correct the logged frame times to align with the Timeline clock
    framePulseFrameNumbers = np.arange(0, len(framePulseTimes) * 200, 200)
    for iPulse in range(len(framePulseTimes)):
        # at each pulse calculate how much the systems have gone out of sync
        # and correct the next 200 frame times in loggedFrameTimes
        tlTimeOfPulse = framePulseTimes[iPulse]
        eyecamTimeOfPulse = loggedFrameTimes[framePulseFrameNumbers[iPulse]]
        driftAtPulse = tlTimeOfPulse - eyecamTimeOfPulse
        # corrected logged times
        if iPulse < len(framePulseTimes)-1:
            loggedFrameTimes[framePulseFrameNumbers[iPulse]:framePulseFrameNumbers[iPulse]+200] += driftAtPulse
        else:
            loggedFrameTimes[framePulseFrameNumbers[iPulse]:] += driftAtPulse

    # # define frames we want to know the times of
    # allFrameNumbers = np.arange(1, eye_frameCount+1)
    # f = interpolate.interp1d(framePulseFrameNumbers, loggedFrameTimes, fill_value = "extrapolate")  # linear interpolation with extrapolation
    # # interpolate / extrapolate missing values'
    # allFrameTimes = f(allFrameNumbers)

    frameRate = 1/np.median(np.diff(loggedFrameTimes))
    print(f"Detected eye cam frame rate = {frameRate}Hz")

    # save vector of eye cam frame times
    np.save(os.path.join(exp_dir_processed_recordings,'eye_frame_times.npy'),loggedFrameTimes)

    print('Done without errors')

# for debugging:
def main():
    # expID
    expID = '2023-02-28_11_ESMT116'
    expID = '2024-07-08_01_TEST'
    # user ID to use to place processed data
    userID = 'adamranson'
    preprocess_cam_run(userID, expID)

if __name__ == "__main__":
    main()