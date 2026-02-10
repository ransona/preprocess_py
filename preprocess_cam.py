import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.io import loadmat
import organise_paths
import pickle
from datetime import datetime

def preprocess_cam_run(userID, expID):
    debug_mode = True
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
        print('Number of frame times = ' + str(len(eye_frameTimes)))

    # Find the index of the 'EyeCamera' channel in the Timeline data
    camIdx = np.where(np.isin(tl_chNames, 'EyeCamera'))[0][0]
    # Extract the camera pulse trace and the corresponding frame pulse times
    camPulseTrace = (tl_daqData[:, camIdx] > 2.5).astype(int)
    # Timing signal should alternate every 100 frames, first transition at frame 100
    alternationTimes = tl_time[np.where(np.diff(camPulseTrace) != 0)]

    # Use all alternating edges (100-frame spacing) to maximize timing data.
    # Each detected edge in Timeline should correspond to one frame every 100 frames
    # in the eye-camera frame-time stream.
    framePulseTimes = alternationTimes

    print('Number of camera timing edges detected in timeline = ' + str(len(framePulseTimes)))
    print('Frames per timing edge = ' + str(len(eye_frameTimes)/len(framePulseTimes)))

    # Do a quality check on timing edge intervals
    # One edge comes every 100 frames acquired
    # @ 20fps this is 5s - should be less than 5s (i.e. better than 20fps)
    # @ 50fps this is 2s - should be more than 2s (i.e. worse than 50fps)
    if np.min(np.diff(framePulseTimes)) < 2:
        if debug_mode:
            plt.figure()
            if len(tl_time) > 100000:
                plt.plot(tl_time[:100000], tl_daqData[:100000, camIdx])
            else:
                plt.plot(tl_time[0:], tl_daqData[0:, camIdx])
            plt.title('Eye camera timing pulses (ch' + str(camIdx) + ' of DAQ)')
            plt.xlabel('Time (secs)')
            plt.ylabel('Voltage (volts)')
        else:
            print('The timing pulses on the eye camera look faulty')
            raise Exception('The timing pulses on the eye camera look faulty - turn on debug mode to see the figure')

    # Detect frame<->timing-edge phase only for older experiments.
    # For experiments on/after 2026-02-10, use the fixed expected first edge at frame 100.
    expDate = datetime.strptime(expID.split('_')[0], '%Y-%m-%d').date()
    phaseDetectionCutoffDate = datetime.strptime('2026-02-10', '%Y-%m-%d').date()
    runPhaseDetection = expDate < phaseDetectionCutoffDate

    # Phase detection goal:
    # find which frame index (0..99) corresponds to the first detected timing edge.
    # Candidate matched edge-frame indices are: phase, phase+100, phase+200, ...
    #
    # Matching metric:
    # compare jitter patterns in inter-edge intervals between Timeline and eye camera.
    # Jitter is median-centered interval sequence, which removes constant rate offset.
    nPulsesForAnalysis = len(framePulseTimes)
    framePulseTimesAnalysis = framePulseTimes[:nPulsesForAnalysis]
    print(f'Using first {nPulsesForAnalysis} timing pulses for phase analysis')

    if len(framePulseTimesAnalysis) < 2:
        raise Exception('Need at least 2 timing pulses to perform jitter-based phase matching')

    if runPhaseDetection:
        bestPhase = None
        bestCorr = -np.inf
        phaseMetrics = []
        for phase in range(100):
            # Build the list of frame indices that would correspond to timing edges
            # under this candidate phase.
            candidatePulseFrameNumbers = phase + 100 * np.arange(len(framePulseTimesAnalysis))
            nComparable = np.sum(candidatePulseFrameNumbers < len(eye_frameTimes))
            if nComparable < 3:
                continue
            candidatePulseFrameNumbers = candidatePulseFrameNumbers[:nComparable]
            candidateFramePulseTimes = framePulseTimesAnalysis[:nComparable]

            # Video-side interval jitter for the same number of candidate edges.
            candidatePulseTimes = eye_frameTimes[candidatePulseFrameNumbers]
            candidatePulseIntervals = np.diff(candidatePulseTimes)
            candidateIntervalJitter = candidatePulseIntervals - np.median(candidatePulseIntervals)
            tlPulseIntervals = np.diff(candidateFramePulseTimes)
            tlIntervalJitter = tlPulseIntervals - np.median(tlPulseIntervals)

            if np.std(candidateIntervalJitter) == 0 or np.std(tlIntervalJitter) == 0:
                corr = -np.inf
            else:
                corr = np.corrcoef(candidateIntervalJitter, tlIntervalJitter)[0, 1]

            phaseMetrics.append({
                'phase': phase,
                'corr': corr
            })

            # Prefer highest jitter correlation.
            if corr > bestCorr:
                bestCorr = corr
                bestPhase = phase

        if bestPhase is None:
            raise Exception(
                'Could not match timeline timing edges to 100-frame-spaced eye frame times'
            )

        print(
            f"Best 100-frame edge phase in eye_frameTimes = {bestPhase} "
            f"(jitter corr {bestCorr:.4f})"
        )

        if debug_mode:
            # Plot histogram of correlation values across all phase offsets.
            corrVals = np.array([m['corr'] for m in phaseMetrics], dtype=float)
            finiteCorrVals = corrVals[np.isfinite(corrVals)]

            fig_corr, ax_corr = plt.subplots(1, 1, figsize=(8, 4))
            ax_corr.hist(finiteCorrVals, bins=30, color='steelblue', edgecolor='black')
            ax_corr.axvline(bestCorr, color='red', linestyle='--', linewidth=1, label='Best corr')
            ax_corr.set_xlabel('Correlation')
            ax_corr.set_ylabel('Count')
            ax_corr.set_title('Distribution of correlations across offsets')
            ax_corr.legend()
            plt.tight_layout()
            plt.show()
    else:
        bestPhase = 100
        print(
            f'Experiment date {expDate} is on/after {phaseDetectionCutoffDate}; '
            f'using fixed 100-frame edge phase = {bestPhase}'
        )

    # Initial alignment step:
    # 1) Re-zero eye camera times so matched edge frame is at t=0.
    # 2) Shift to place that edge at the first Timeline edge time.
    # After this, loggedFrameTimes[bestPhase] == framePulseTimes[0].
    loggedFrameTimes = eye_frameTimes - eye_frameTimes[bestPhase]
    loggedFrameTimes = loggedFrameTimes + framePulseTimes[0]

    # Drift correction step (piecewise-constant):
    # At each timing edge, compute mismatch between Timeline and current logged time
    # at the matched edge frame, then add that mismatch to the following 100-frame block.
    #
    # This enforces exact agreement at every detected edge but does not stretch/compress
    # time within each 100-frame block.
    nPulsesComparable = min(
        len(framePulseTimes),
        ((len(eye_frameTimes) - 1 - bestPhase) // 100) + 1
    )
    framePulseTimesUsed = framePulseTimes[:nPulsesComparable]
    framePulseFrameNumbers = bestPhase + 100 * np.arange(nPulsesComparable)

    for iPulse in range(nPulsesComparable):
        # at each edge calculate how much the systems have gone out of sync
        # and correct the next 100 frame times in loggedFrameTimes
        tlTimeOfPulse = framePulseTimesUsed[iPulse]
        eyecamTimeOfPulse = loggedFrameTimes[framePulseFrameNumbers[iPulse]]
        driftAtPulse = tlTimeOfPulse - eyecamTimeOfPulse
        # corrected logged times
        if iPulse < nPulsesComparable - 1:
            loggedFrameTimes[framePulseFrameNumbers[iPulse]:framePulseFrameNumbers[iPulse]+100] += driftAtPulse
        else:
            loggedFrameTimes[framePulseFrameNumbers[iPulse]:] += driftAtPulse

    frameRate = 1/np.median(np.diff(loggedFrameTimes))
    print(f"Detected eye cam frame rate = {frameRate}Hz")

    # save vector of eye cam frame times
    np.save(os.path.join(exp_dir_processed_recordings,'eye_frame_times.npy'),loggedFrameTimes)

    print('Done without errors')

# for debugging:
def main():
    userID = 'rubencorreia'
    expIDs = [
        '2026-01-19_06_ESRC023']

    for expID in expIDs:
        preprocess_cam_run(userID, expID)

if __name__ == "__main__":
    main()
