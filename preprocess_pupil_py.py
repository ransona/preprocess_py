import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import polygon
from circle_fit import taubinSVD as circle_fit
import time
import os
import pandas as pd
import pickle
import organise_paths

def preprocess_pupil_run(userID, expID):
    print('Starting preprocess_pupil_run...')
    animalID, remote_repository_root, \
    processed_root, exp_dir_processed, \
        exp_dir_raw = organise_paths.find_paths(userID, expID)
    exp_dir_processed_recordings = os.path.join(processed_root, animalID, expID,'recordings')

    if not os.path.exists(exp_dir_processed_recordings):
        os.mkdir(exp_dir_processed_recordings)

    displayOn = False
    displayInterval = 100

    if displayOn:
        f = 0 #figure()

    print('Starting ' + expID)

    dlc_filenames = [expID + '_eye1_leftDLC_resnet50_Trial_newMay19shuffle1_1030000.csv',
                    expID + '_eye1_rightDLC_resnet50_Trial_newMay19shuffle1_1030000.csv']

    vid_filenames = [expID + '_eye1_left.avi',
                    expID + '_eye1_right.avi']


    for iVid in range(0, len(dlc_filenames)):
        print()
        print('Starting video ' + str(iVid))
        videoPath = os.path.join(exp_dir_processed, vid_filenames[iVid])
        v = cv2.VideoCapture(videoPath)

        if not v.isOpened():
            raise Exception('Error: Eye video file not found')

        # read the csv deeplabcut output file
        dlc_data = pd.read_csv(os.path.join(exp_dir_processed, dlc_filenames[iVid]), delimiter=',',skiprows=[0,1,2],header=None)

        # remove first column
        z = dlc_data.iloc[1,1]
        eyeX = dlc_data.iloc[:,[25,28,31,34]].values
        eyeY = dlc_data.loc[:,[26,29,32,35]].values
        pupilX = dlc_data.loc[:,1:22:3].values
        pupilY = dlc_data.loc[:,2:23:3].values
        # get minimum of eye x and eye y confidence from dlc
        # eye x and eye y are always needed as a minimum to process a frame so
        # we ensure below that these coordinates all have confidence > 0.8
        eyeMinConfid = np.min(dlc_data.loc[:,26::3], axis=1)
        ret, firstFrame = v.read()
        frameSize = np.squeeze(firstFrame[:,:,0]).shape
        
        # choose approximate eye area - points outside this will be considered
        # invalid
        roiLeft = np.median(eyeX[:,2],0).astype(int)
        roiTop = np.median(eyeY[:,1],0).astype(int)
        roiWidth = (np.median(eyeX[:,0],0) - roiLeft).astype(int)
        roiHeight = (np.median(eyeY[:,3]) - roiTop).astype(int)
        padding = int((roiWidth * 0.75))

        # for debugging the assumed eye position:
        # plt.figure()
        # plt.imshow(firstFrame)
        # rect = plt.Rectangle((roiLeft-padding,roiTop-padding), roiWidth+padding*2, roiHeight+padding*2, edgecolor='r', fill=False)
        # ax = plt.gca()
        # ax.add_patch(rect)
        # plt.show()
        # plt.close()
        #############
        validRegionMask = np.zeros(frameSize)
        topLimit = roiTop-padding
        bottomLimit = roiTop+roiHeight+padding
        leftLimit = roiLeft-padding
        rightLimit = roiLeft+roiWidth+padding
        if topLimit < 0: topLimit = 0
        if bottomLimit > frameSize[0]: bottomLimit = frameSize[0]
        if leftLimit < 0: leftLimit = 0
        if rightLimit > frameSize[1]: rightLimit = frameSize[1]
        validRegionMask[topLimit:bottomLimit,leftLimit:rightLimit] = 1
        #plt.figure()
        #plt.imshow(validRegionMask)
        # calc some average values for eye to be used for QC later
        eyeWidth = (np.median(eyeX[:,0])-np.median(eyeX[:,2]))
        # clip coordinates to frame size
        eyeX[eyeX>frameSize[1]] = frameSize[1]
        eyeY[eyeY>frameSize[0]] = frameSize[0]
        pupilX[pupilX>frameSize[1]] = frameSize[1]
        pupilY[pupilY>frameSize[0]] = frameSize[0]
        eyeX[eyeX<0] = 0
        eyeY[eyeY<0] =0
        pupilX[pupilX<0] = 0
        pupilY[pupilY<0] = 0
        
        # create dict to output eye data to
        eyeDat = {}
        eyeDat['x'] = []
        eyeDat['y'] = []
        eyeDat['radius'] = []
        eyeDat['qc'] = []

        lastFrame = time.time()
        for iFrame in range(dlc_data.shape[0]):
            # do QC to make sure the eye corners have been well detected
            pointsValid = validRegionMask[
                np.ceil(eyeY[iFrame]).astype(int),
                np.ceil(eyeX[iFrame]).astype(int)
            ]
            # check spacing of left and right corners is about right compared to
            # median of whole recording
            cornerDistanceDiff = np.abs((eyeX[iFrame,0]-eyeX[iFrame,2])-eyeWidth)/eyeWidth
            # check top and bottom lid mid points are around halfway between eye
            # corners in x direction
            min_corner_middle_distance = np.min(np.abs([
                eyeX[iFrame,2]-eyeX[iFrame,1],
                eyeX[iFrame,2]-eyeX[iFrame,3],
                eyeX[iFrame,0]-eyeX[iFrame,1],
                eyeX[iFrame,0]-eyeX[iFrame,3]
            ]))

            # confirm all 4 eye corners are within expected region AND that the smallest of 
            # the mid-eyelid to lat-eye corner distances is more than 30% of eye width (it
            # should be about 50%)
            if np.min(pointsValid) == 1 and (min_corner_middle_distance / eyeWidth) > 0.30:

                # fit two parabolas - one for each eye lid
                # points:
                # 1 = lateral / 2 = sup / 3 = medial / 4 = inf
                topLid = np.polyfit(eyeX[iFrame, [0, 1, 2]], eyeY[iFrame, [0, 1, 2]], 2)
                botLid = np.polyfit(eyeX[iFrame, [0, 3, 2]], eyeY[iFrame, [0, 3, 2]], 2)

                # generate points
                xVals = np.linspace(eyeX[iFrame, 0], eyeX[iFrame, 2])
                # for upper lid
                yVals = topLid[0] * xVals**2 + topLid[1] * xVals + topLid[2]
                # for lower lid
                yVals = np.concatenate([yVals, botLid[0] * np.flipud(xVals)**2 + botLid[1] * np.flipud(xVals) + botLid[2]])
                xVals = np.concatenate([xVals, np.flipud(xVals)])
                # check if y values
                # make a poly mask using the points
                rr, cc = polygon(yVals, xVals)
                eyeMask = np.zeros(frameSize)
                eyeMask[rr, cc] = 1
                # plt.figure()
                # plt.imshow(eyeMask)
                # check if each pupil point is in the eye mask and exclude it if not
                # to do this convert x y coordinates 
                pupilIdx = np.ravel_multi_index([[pupilY[iFrame].astype(int)], [pupilX[iFrame].astype(int)]], frameSize)
                inEye = eyeMask.flatten()[pupilIdx][0].astype(bool)
                # fit a circle to those pupil points within the eye
                xpoints = pupilX[iFrame, inEye]
                ypoints = pupilY[iFrame, inEye]
                allpoints = np.concatenate((xpoints[np.newaxis,:],ypoints[np.newaxis,:]),axis=0).T
                if np.sum(inEye) > 2:
                    xCenter, yCenter, radius, _ = circle_fit(allpoints)
                else:
                    # not enough points to fit circle
                    xCenter = np.nan
                    yCenter = np.nan
                    radius = np.nan

                eyeDat['x'].append(xCenter.astype(int))
                eyeDat['y'].append(yCenter.astype(int))
                eyeDat['radius'].append(radius.astype(int))

                # default to quality control passed
                eyeDat['qc'].append(0)

                if np.sum(inEye) < 2:
                    eyeDat['qc'][iFrame] = 2  # indicates QC failed due to pupil fit

                if iFrame==0:
                    # initialise data structure
                    eyeDat['topLid']=(topLid[np.newaxis,:])
                    eyeDat['botLid']=(botLid[np.newaxis,:])
                    eyeDat['inEye']=(inEye[np.newaxis,:])
                else:
                    eyeDat['topLid']=np.concatenate((eyeDat['topLid'],topLid[np.newaxis,:]),axis=0)
                    eyeDat['botLid']=np.concatenate((eyeDat['botLid'],botLid[np.newaxis,:]),axis=0)
                    eyeDat['inEye']=np.concatenate((eyeDat['inEye'],inEye[np.newaxis,:]),axis=0)

                if iFrame % displayInterval == 0:
                    print(f'{iFrame}/{dlc_data.shape[0]} - {iFrame/dlc_data.shape[0]*100:.2f}% complete'+f' Frame rate = {(1/(time.time()-lastFrame))*displayInterval:.2f}', end='\r')
                    #print(f'Frame rate = {(1/(time.time()-lastFrame))*displayInterval:.2f}')
                    lastFrame = time.time()

            else:
                # frame has failed QC
                eyeDat['qc'].append(0)
                eyeDat['x'].append(np.nan)
                eyeDat['y'].append(np.nan)
                eyeDat['radius'].append(np.nan)
                if iFrame==0:
                    # initialise data structure
                    eyeDat['topLid']=np.nan
                    eyeDat['botLid']=np.nan
                    eyeDat['inEye']=np.nan
                else:
                    # add a row of nans of the right shape (width)
                    eyeDat['topLid']=np.concatenate((eyeDat['topLid'],np.full((1,eyeDat['topLid'].shape[1]),np.nan)),axis=0)
                    eyeDat['botLid']=np.concatenate((eyeDat['botLid'],np.full((1,eyeDat['botLid'].shape[1]),np.nan)),axis=0)
                    eyeDat['inEye']=np.concatenate((eyeDat['inEye'],np.full((1,eyeDat['inEye'].shape[1]),np.nan)),axis=0)

            if displayOn:
                if iFrame == 0:
                    fig = plt.figure()

                if iFrame % displayInterval == 0:

                    if eyeDat['qc'][iFrame] == 0:
                        # quality control passed

                        # set to read next frame
                        v.set(cv2.CAP_PROP_POS_FRAMES, iFrame)
                        # read the frame at the current position
                        ret, currentFrame = v.read()
                        currentFrame = np.squeeze(currentFrame[:, :, 1])
                        # all plotting
                        plt.clf()
                        plt.imshow(currentFrame, cmap='gray')
                        # plot all coordinates
                        # plot eye
                        plt.plot(xVals, yVals, 'y')
                        # plot valid pupil points
                        plt.scatter(pupilX[iFrame, inEye], pupilY[iFrame, inEye], c='g')
                        # invalid
                        plt.scatter(pupilX[iFrame, ~inEye], pupilY[iFrame, ~inEye], c='r')
                        # draw pupil circle
                        plt.gca().add_artist(plt.Circle((xCenter, yCenter), radius, fill=False))
                        plt.show()
                        fig.canvas.draw()
                        fig.canvas.flush_events()
                    else:
                        # quality control NOT passed
                        plt.figure()
                        currentFrame = readVideoIndex(v, iFrame)
                        currentFrame = np.squeeze(currentFrame[:, :, 1])
                        # all plotting
                        plt.clf()
                        plt.imshow(currentFrame, cmap='gray')
                        # plot all coordinates
                        plt.scatter(pupilX[iFrame, :], pupilY[iFrame, :], c='g')
                        # plot valid pupil points
                        plt.scatter(pupilX[iFrame, :], pupilY[iFrame, :], c='g')
                        # eye points
                        plt.scatter(eyeX[iFrame, :], eyeY[iFrame, :])
                        plt.show()

        # do some further quality control
        # remove points where pupil looks dodgy
        # validFrames = np.ones(len(videoTiming['pupil']['x']))
        # validFrames[np.isnan(videoTiming['pupil']['x'])] = 0
        # validFrames[videoTiming['pupil']['radius'] > 100] = 0
        # validFrames[videoTiming['pupil']['x'] > np.median(videoTiming['pupil']['x'][validFrames==1])+100] = 0
        # validFrames[videoTiming['pupil']['x'] < np.median(videoTiming['pupil']['x'][validFrames==1])-100] = 0
        # validFrames[videoTiming['pupil']['y'] > np.median(videoTiming['pupil']['y'][validFrames==1])+100] = 0
        # validFrames[videoTiming['pupil']['y'] < np.median(videoTiming['pupil']['y'][validFrames==1])-100] = 0
        # videoTiming['pupil']['x'][validFrames==0] = np.nan
        # videoTiming['pupil']['y'][validFrames==0] = np.nan
        # videoTiming['pupil']['radius'][validFrames==0] = np.nan

        # calculate pupil velocity
        xdiffs = np.diff(eyeDat['x'])
        ydiffs = np.diff(eyeDat['y'])
        eucla_diff = np.sqrt(xdiffs**2 + ydiffs**2)
        eyeDat['velocity'] = np.convolve(eucla_diff, np.ones(10), 'same')
        eyeDat['velocity'] = np.append(eyeDat['velocity'], eyeDat['velocity'][-1])

        # store detected eye details with timeline timestamps
        # load video timestamps
        loggedFrameTimes = np.load(os.path.join(exp_dir_processed_recordings,'eye_frame_times.npy'))
        # resample to 10Hz constant rate
        newTimeVector = np.arange(round(loggedFrameTimes[0]), round(loggedFrameTimes[-1]), 0.1)
        frameVector = np.arange(0,len(eyeDat['x']))
        eyeDat2 = {}
        eyeDat2['t'] = newTimeVector
        eyeDat2['x'] = np.interp(newTimeVector, loggedFrameTimes, eyeDat['x'])
        eyeDat2['y'] = np.interp(newTimeVector, loggedFrameTimes, eyeDat['y'])
        eyeDat2['radius'] = np.interp(newTimeVector, loggedFrameTimes, eyeDat['radius'])
        eyeDat2['velocity'] = np.interp(newTimeVector, loggedFrameTimes, eyeDat['velocity'])
        eyeDat2['qc'] = np.interp(newTimeVector, loggedFrameTimes, eyeDat['qc'])
        eyeDat2['frame'] = np.round(np.interp(newTimeVector, loggedFrameTimes, frameVector))
        if iVid == 0:
            pickle_out = open(os.path.join(exp_dir_processed_recordings,'dlcEyeLeft.pickle'),"wb")
            pickle.dump(eyeDat, pickle_out)
            pickle_out.close()
            pickle_out = open(os.path.join(exp_dir_processed_recordings,'dlcEyeLeft_resampled.pickle'),"wb")
            pickle.dump(eyeDat2, pickle_out)
            pickle_out.close()
        else:
            pickle_out = open(os.path.join(exp_dir_processed_recordings,'dlcEyeRight.pickle'),"wb")
            pickle.dump(eyeDat, pickle_out)
            pickle_out.close()
            pickle_out = open(os.path.join(exp_dir_processed_recordings,'dlcEyeRight_resampled.pickle'),"wb")
            pickle.dump(eyeDat2, pickle_out)
            pickle_out.close()           
    print()
    print('Done without errors')

# for debugging:
def main():
    # expID
    expID = '2023-02-28_11_ESMT116'
    # user ID to use to place processed data
    userID = 'adamranson'
    preprocess_pupil_run(userID, expID)

if __name__ == "__main__":
    main()