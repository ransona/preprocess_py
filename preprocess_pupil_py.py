import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import polygon
from circle_fit import circle_fit as circlefit
import time
import os

expID = '2022-01-21_04_ESPM039'
dataRoot = 'D:/data'
animalID = expID[14:]

expRoot = 'G:\.shortcut-targets-by-id\18E8Ww5qCgzn27qk_LrsR3R6wMweRdJ_Z\AR_RRP\ESPM039\2022-01-21_04_ESPM039'

displayOn = True
displayInterval = 100

if displayOn:
    f = 0 #figure()

print('Starting ' + expID)

dlc_filenames = [expID + '_eye1_leftDLC_resnet50_Trial_newMay19shuffle1_1030000.csv',
                 expID + '_eye1_rightDLC_resnet50_Trial_newMay19shuffle1_1030000.csv']

vid_filenames = [expID + '_eye1_left.avi',
                 expID + '_eye1_right.avi']


for iVid in range(1, len(dlc_filenames)):

    videoPath = os.path.join(expRoot, vid_filenames[iVid])
    v = cv2.VideoCapture(videoPath)

    if not v.isOpened():
        raise Exception('Error: Eye video file not found')

    # read the csv deeplabcut output file
    dlc_data = np.loadtxt(os.path.join(expRoot, dlc_filenames[iVid]), delimiter=',')
    # remove first column
    dlc_data = dlc_data[:,1:]
    eyeX = dlc_data[:,24::3]
    eyeY = dlc_data[:,25::3]
    pupilX = dlc_data[:,0:24:3]
    pupilY = dlc_data[:,1:24:3]
    # get minimum of eye x and eye y confidence from dlc
    # eye x and eye y are always needed as a minimum to process a frame so
    # we ensure below that these coordinates all have confidence > 0.8
    eyeMinConfid = np.min(dlc_data[:,26::3], axis=1)
    
    ret, firstFrame = v.read()
    frameSize = np.squeeze(firstFrame[:,:,0]).shape
    
    # choose approximate eye area - points outside this will be considered
    # invalid
    plt.imshow(firstFrame)
    roiLeft = np.median(eyeX[:,2])
    roiTop = np.median(eyeY[:,1])
    roiWidth = 50 # np.median(eyeX[:,0]) - roiLeft
    roiHeight = 50 # np.median(eyeY[:,3]) - roiTop
    padding = roiWidth * 0.75
    rect = plt.Rectangle((roiLeft-padding,roiTop-padding), roiWidth+padding*2, roiHeight+padding*2, edgecolor='r', fill=False)
    ax = plt.gca()
    ax.add_patch(rect)
    plt.show()
    plt.close()
    validRegionMask = np.zeros(frameSize, dtype=bool)
    validRegionMask[roiTop-padding:roiTop+roiHeight+padding, roiLeft-padding:roiLeft+roiWidth+padding] = True
    # calc some average values for eye to be used for QC later
    eyeWidth = (np.median(eyeX[:,0])-np.median(eyeX[:,2]))
    # clip coordinates to frame size
    eyeX[eyeX>frameSize[1]] = frameSize[1]
    eyeY[eyeY>frameSize[0]] = frameSize[0]
    pupilX[pupilX>frameSize[1]] = frameSize[1]
    pupilY[pupilY>frameSize[0]] = frameSize[0]
    eyeX[eyeX<1] = 1
    eyeY[eyeY<1] = 1
    pupilX[pupilX<1] = 1
    pupilY[pupilY<1] = 1
    
    eyeDat = []
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

        if np.min(pointsValid) == 1 and (min_corner_middle_distance / eyeWidth) > 0.33:

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
            yVals = np.concatenate([yVals, botLid[0] * np.fliplr(xVals)**2 + botLid[1] * np.fliplr(xVals) + botLid[2]])
            xVals = np.concatenate([xVals, np.fliplr(xVals)])
            # check if y values
            # make a poly mask using the points
            rr, cc = polygon(yVals, xVals)
            eyeMask = np.zeros(frameSize)
            eyeMask[rr, cc] = 1

            # check if each pupil point is in the eye mask and exclude it if not
            pupilIdx = np.ravel_multi_index((np.round(pupilY[iFrame]), np.round(pupilX[iFrame])), frameSize)
            inEye = eyeMask.flatten()[pupilIdx]

            # fit a circle to those pupil points within the eye
            if np.sum(inEye) > 2:
                xCenter, yCenter, radius, _ = circlefit(pupilX[iFrame, inEye], pupilY[iFrame, inEye])
            else:
                # not enough points to fit circle
                xCenter = np.nan
                yCenter = np.nan
                radius = np.nan

            eyeDat.x[iFrame] = xCenter
            eyeDat.y[iFrame] = yCenter

            if np.sum(inEye) < 2:
                eyeDat.qc[iFrame] = 2  # indicates QC failed due to pupil fit
            else:
                eyeDat.qc[iFrame] = 0  # indicates QC passed

            eyeDat.radius[iFrame] = radius
            eyeDat.topLid[iFrame] = topLid
            eyeDat.botLid[iFrame] = botLid
            eyeDat.inEye[iFrame] = inEye

            if iFrame % displayInterval == 0:
                print('#####################')
                print(f'{iFrame}/{dlc_data.shape[0]} - {iFrame/dlc_data.shape[0]*100:.2f}% complete')
                print(f'Frame rate = {1/lastFrameTime:.2f}')
                print('#####################')

        else:
            # frame has failed QC
            eyeDat.x[iFrame] = np.nan
            eyeDat.y[iFrame] = np.nan
            eyeDat.radius[iFrame] = np.nan
            eyeDat.topLid[iFrame] = np.nan
            eyeDat.botLid[iFrame]

    lastFrame = time.time()

    if displayOn:
        if iFrame % displayInterval == 0:
            if eyeDat.qc[iFrame] == 0:
                # quality control passed
                plt.figure()
                currentFrame = readVideoIndex(v, iFrame)
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
    if iVid == 1:
        scipy.io.savemat(os.path.join(expRoot, 'dlcEyeLeft.mat'), {'eyeDat': eyeDat})
    else:
        scipy.io.savemat(os.path.join(expRoot, 'dlcEyeRight.mat'), {'eyeDat': eyeDat})

def circlefit(x, y):
    numPoints = len(x)
    xx = x * x
    yy = y * y
    xy = x * y
    A = np.array([[np.sum(x),  np.sum(y),  numPoints],
                  [np.sum(xy), np.sum(yy), np.sum(y)],
                  [np.sum(xx), np.sum(xy), np.sum(x)]])
    B = np.array([-np.sum(xx + yy),
                  -np.sum(xx * y + yy * y),
                  -np.sum(xx * x + xy * y)])
    a = np.linalg.solve(A, B)
    xCenter = -0.5 * a[0]
    yCenter = -0.5 * a[1]
    radius = np.sqrt((a[0] ** 2 + a[1] ** 2) / 4 - a[2])
    return xCenter, yCenter, radius, a

