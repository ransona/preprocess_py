
import os.path as osp

# Load the data from the file
expRootLocal = "/path/to/expRootLocal"
expID = "my_exp"
eyeMeta1_mat = os.path.join(expRootLocal, f"{expID}_eyeMeta1.mat")
eyeMeta1 = np.loadmat(eyeMeta1_mat)

Timeline = eyeMeta1["Timeline"]
eTrackData = eyeMeta1["eTrackData"]

# Find the index of the camera channel
camIdx = np.where(Timeline["chNames"] == "EyeCamera")[0][0]

# Get the camera pulse trace and frame pulse times
camPulseTrace = Timeline["daqData"][:, camIdx] > 2.5
framePulseTimes = Timeline["time"][np.where(np.diff(camPulseTrace) == 1)[0] + 1]

# Perform a quality check on the frame pulse times
if np.min(np.diff(framePulseTimes)) < 16:
    plt.figure()
    if len(Timeline["time"]) > 100000:
        plt.plot(Timeline["time"][:100000], Timeline["daqData"][:100000, camIdx])
    else:
        plt.plot(Timeline["time"], Timeline["daqData"][:, camIdx])
    plt.title(f"Eye camera timing pulses (ch{camIdx} of DAQ)")
    plt.xlabel("Time (secs)")
    plt.ylabel("Voltage (volts)")
    print("The timing pulses on the eye camera look faulty - see the figure")
    raise ValueError("The timing pulses on the eye camera look faulty - see the figure")

# Adjust the logged frame times to be approximately in timeline time
loggedFrameTimes = eTrackData["frameTimes"] - eTrackData["frameTimes"][0]
loggedFrameTimes = loggedFrameTimes + framePulseTimes[0]

# Periodically correct the logged frame times to the timeline clock
framePulseFrameNumbers = np.arange(0, len(framePulseTimes)*200, 200)

import numpy as np

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
allFrameNumbers = np.arange(1, eTrackData.frameCount+1)
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
