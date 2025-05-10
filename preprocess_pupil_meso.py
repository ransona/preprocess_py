# take dlc pipil output and fits circle to pupil etc
import cProfile
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import polygon
from circle_fit import taubinSVD as circle_fit
from scipy.ndimage import median_filter
import time
import os
import pandas as pd
import pickle
import organise_paths
import shutil
from scipy.ndimage import binary_dilation, binary_erosion
from PIL import Image, ImageDraw

def preprocess_pupil_run(userID, expID):
    print('Starting preprocess_pupil_run...')
    animalID, remote_repository_root, \
    processed_root, exp_dir_processed, \
        exp_dir_raw = organise_paths.find_paths(userID, expID)
    exp_dir_processed_recordings = os.path.join(processed_root, animalID, expID,'recordings')

    if not os.path.exists(exp_dir_processed_recordings):
        os.mkdir(exp_dir_processed_recordings)

    displayOn = False
    displayInterval = 1000

    if displayOn:
        f = 0 #figure()

    print('Starting ' + expID)

    dlc_filenames = [f'{expID}_eye1_leftDLC_Resnet50_EYEMay8shuffle1_snapshot_010.csv',
                    f'{expID}_eye1_rightDLC_Resnet50_EYEMay8shuffle1_snapshot_010.csv']

    vid_filenames = [expID + '_eye1_left.avi',
                    expID + '_eye1_right.avi']


    for iVid in range(0, len(dlc_filenames)):
        print()
        print('Starting video ' + str(iVid))

        videoPath = os.path.join(exp_dir_processed, vid_filenames[iVid])
        # check if cropped videos are in the processed data directory and if not try to copy from the remote repos
        # this can be removed in the future 
        if not os.path.isfile(os.path.join(exp_dir_processed, vid_filenames[iVid])):
            try:
                shutil.copyfile(os.path.join(exp_dir_raw, vid_filenames[iVid]),os.path.join(exp_dir_processed, vid_filenames[iVid]))
            except:
                print('Cropped eye videos not found on server')

        # read the csv deeplabcut output file
        dlc_data = pd.read_csv(os.path.join(exp_dir_processed, dlc_filenames[iVid]), delimiter=',',skiprows=[0,1,2],header=None)

        eyeX = dlc_data.iloc[:,[25,28,31,34]].values
        eyeY = dlc_data.loc[:,[26,29,32,35]].values
        pupilX = dlc_data.loc[:,1:22:3].values
        pupilY = dlc_data.loc[:,2:23:3].values
        # get minimum of eye x and eye y confidence from dlc
        # apply a median filter accross time to remove random blips 
        eyeX = median_filter(eyeX,[3,1])
        eyeY = median_filter(eyeY,[3,1])
        pupilX = median_filter(pupilX,[3,1])
        pupilY = median_filter(pupilY,[3,1])
        # eye x and eye y are always needed as a minimum to process a frame so
        # we ensure below that these coordinates all have confidence > 0.8
        eyeMinConfid = np.min(dlc_data.loc[:,26::3], axis=1)
        v = cv2.VideoCapture(videoPath)
        ret, firstFrame = v.read()
        firstFrame = firstFrame[:,:,0]
        frameSize = firstFrame.shape
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
        # plt.figure()
        # plt.imshow(validRegionMask)
        # calc some average values for eye to be used for QC later
        eyeWidth = (np.median(eyeX[:,0])-np.median(eyeX[:,2]))
        # clip coordinates to frame size
        eyeX[eyeX>frameSize[1]-1] = frameSize[1]-1
        eyeY[eyeY>frameSize[0]-1] = frameSize[0]-1
        pupilX[pupilX>frameSize[1]-1] = frameSize[1]-1
        pupilY[pupilY>frameSize[0]-1] = frameSize[0]-1
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
        eyeDat['eye_lid_x'] = np.full((dlc_data.shape[0],40),np.nan)
        eyeDat['eye_lid_y'] = np.full((dlc_data.shape[0],40),np.nan)
        eyeDat['eyeX'] = eyeX
        eyeDat['eyeY'] = eyeY
        eyeDat['pupilX'] = pupilX
        eyeDat['pupilY'] = pupilY

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
            if eyeWidth == 0:
                z=0

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
            if np.min(pointsValid) == 1: # and (min_corner_middle_distance / eyeWidth) > 0.30:
                # Extract and expand polygon vertices
                x_coords = eyeX[iFrame, [0, 1, 2, 3, 0]] + np.array([40, 0, -40, 0, 40])
                y_coords = eyeY[iFrame, [0, 1, 2, 3, 0]] + np.array([0, -40, 0, +40, 0])

                # Create mask same size as firstFrame
                height, width = firstFrame.shape
                mask_img = Image.new('L', (width, height), 0)
                draw = ImageDraw.Draw(mask_img)

                # Draw filled polygon
                polygon = list(zip(map(float, x_coords), map(float, y_coords)))
                draw.polygon(polygon, outline=1, fill=1)
                polygon_mask = np.array(mask_img, dtype=bool)

                # Extract perimeter using binary difference of mask's edges
                edges_y, edges_x = np.where(np.logical_xor(polygon_mask, binary_erosion(polygon_mask)))
                coords = np.column_stack((edges_y, edges_x))
                n = len(coords)

                # Sample 40 perimeter points
                if n >= 40:
                    idx = np.linspace(0, n - 1, 40, dtype=int)
                    sampled = coords[idx]
                else:
                    reps = int(np.ceil(40 / n))
                    tiled = np.tile(coords, (reps, 1))
                    sampled = tiled[:40]

                y_perim, x_perim = sampled[:, 0], sampled[:, 1]

                eyeDat['eye_lid_x'][iFrame,:] = x_perim
                eyeDat['eye_lid_y'][iFrame,:] = y_perim

                eyeMask = polygon_mask.astype(np.uint8)

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
                    eyeDat['x'].append(xCenter.astype(int))
                    eyeDat['y'].append(yCenter.astype(int))
                    eyeDat['radius'].append(radius.astype(int))
                else:
                    # not enough points to fit circle
                    xCenter = np.nan
                    yCenter = np.nan
                    radius = np.nan
                    eyeDat['x'].append(xCenter)
                    eyeDat['y'].append(yCenter)
                    eyeDat['radius'].append(radius)

                def sample_top_bottom_arcs(perimeter_mask, n_each=20):
                    # 1) get all (row, col) perimeter coords
                    coords = np.column_stack(np.nonzero(perimeter_mask))
                    # 2) compute centroid
                    centroid = coords.mean(axis=0)   # [centroid_row, centroid_col]

                    # 3) for each point compute a “dy, dx” vector from centroid,
                    #    and its angle in Cartesian coords (y inverted so up is +dy)
                    dy = centroid[0] - coords[:, 0]
                    dx = coords[:, 1] - centroid[1]
                    angles = np.arctan2(dy, dx)      # range (–π, +π]

                    # 4) split into top (dy ≥ 0) and bottom (dy < 0)
                    top_mask    = dy >= 0
                    bottom_mask = dy <  0

                    top_coords    = coords[top_mask]
                    top_angles    = angles[top_mask]
                    bottom_coords = coords[bottom_mask]
                    bottom_angles = angles[bottom_mask]

                    # 5) sort each half by angle so they run smoothly along the arc
                    #    – Top arc: angles go from 0 (right) up to π (left), so reverse for left→right
                    top_sorted    = top_coords[np.argsort(top_angles)[::-1]]
                    #    – Bottom arc: angles go from –π (left) up to 0 (right)
                    bottom_sorted = bottom_coords[np.argsort(bottom_angles)]

                    # 6) resample exactly n_each points from each sorted list
                    def resample(arr):
                        m = len(arr)
                        if m >= n_each:
                            idx = np.linspace(0, m-1, n_each, dtype=int)
                            return arr[idx]
                        else:
                            # if too few, tile and truncate
                            reps = int(np.ceil(n_each / m))
                            tiled = np.tile(arr, (reps, 1))
                            return tiled[:n_each]

                    top20    = resample(top_sorted)
                    bottom20 = resample(bottom_sorted)

                    # 7) return as x,y each
                    y_top,    x_top    = top20[:, 0],    top20[:, 1]
                    y_bottom, x_bottom = bottom20[:, 0], bottom20[:, 1]
                    return x_top, y_top, x_bottom, y_bottom

                #x_top, y_top, x_bottom, y_bottom = sample_top_bottom_arcs(perimeter_mask)

                if displayOn:
                    # plot the fitted circle
                    plt.figure()
                    plt.imshow(firstFrame, cmap='gray')
                    ax = plt.gca()
                    circle = plt.Circle((xCenter, yCenter), radius, color='r', fill=False)
                    ax.add_patch(circle)
                    ax.set_aspect('equal')   # ensure circle isn’t distorted
                    plt.scatter(x_perim, y_perim, s=1, c='b', label='Perimeter')
                    #plt.scatter(x_top, y_top, s=1, c='r', label='Top arc')
                    #plt.scatter(x_bottom, y_bottom, s=1, c='b', label='Bottom arc')
                    # plt.plot(x_coords, y_coords, 'g-o', label='Original polygon')
                    # plt.scatter(x_perim, y_perim, s=1, c='b', label='Dilated perimeter')
                    plt.legend()
                    plt.show()


                # default to quality control passed
                eyeDat['qc'].append(0)

                if np.sum(inEye) < 2:
                    eyeDat['qc'][iFrame] = 2  # indicates QC failed due to pupil fit

                if iFrame==0:
                    # initialise data structure
                    eyeDat['inEye']=(inEye[np.newaxis,:])
                else:
                    eyeDat['inEye']=np.concatenate((eyeDat['inEye'],inEye[np.newaxis,:]),axis=0)

                if iFrame % displayInterval == 0:
                    print(f'{iFrame}/{dlc_data.shape[0]} - {iFrame/dlc_data.shape[0]*100:.2f}% complete'+f' Frame rate = {(1/(time.time()-lastFrame))*displayInterval:.2f}', end='\r')
                    #print(f'Frame rate = {(1/(time.time()-lastFrame))*displayInterval:.2f}')
                    lastFrame = time.time()

            else:
                # frame has failed QC
                eyeDat['qc'].append(1)
                eyeDat['x'].append(np.nan)
                eyeDat['y'].append(np.nan)
                eyeDat['radius'].append(np.nan)
                if iFrame==0:
                    # initialise data structure
                    eyeDat['inEye']=np.full([1,8],np.nan)
                else:
                    # add a row of nans of the right shape (width)
                    try:
                        eyeDat['inEye']=np.concatenate((eyeDat['inEye'],np.full((1,eyeDat['inEye'].shape[1]),np.full([1,8],np.nan))),axis=0)
                    except:
                        z = 0

        # convert lists to arrays
        eyeDat['x'] = np.array(eyeDat['x'])
        eyeDat['y'] = np.array(eyeDat['y']) 
        eyeDat['radius'] = np.array(eyeDat['radius']) 
        eyeDat['qc'] = np.array(eyeDat['qc']) 

        # filter x / y / rad
        eyeDat['x'] = median_filter(eyeDat['x'],[9])
        eyeDat['y'] = median_filter(eyeDat['y'],[9])
        eyeDat['radius'] = median_filter(eyeDat['radius'],[9])
        
        # calculate pupil velocity
        xdiffs = np.diff(eyeDat['x'])
        ydiffs = np.diff(eyeDat['y'])
        eucla_diff = np.sqrt(xdiffs**2 + ydiffs**2)
        eyeDat['velocity'] = np.convolve(eucla_diff, np.ones(10), 'same')
        eyeDat['velocity'] = np.append(eyeDat['velocity'], eyeDat['velocity'][-1])
        eyeDat['velocity'] = np.array(eyeDat['velocity']) 

        if iVid == 0:
            pickle_out = open(os.path.join(exp_dir_processed_recordings,'dlcEyeLeft.pickle'),"wb")
            pickle.dump(eyeDat, pickle_out)
            pickle_out.close()
        else:
            pickle_out = open(os.path.join(exp_dir_processed_recordings,'dlcEyeRight.pickle'),"wb")
            pickle.dump(eyeDat, pickle_out)
            pickle_out.close()
        print(f'{iFrame}/{dlc_data.shape[0]-1} - {(iFrame+1)/dlc_data.shape[0]*100:.2f}% complete'+f' Frame rate = {(1/(time.time()-lastFrame))*displayInterval:.2f}', end='\r')        
    print()
    print('Done without errors')

# for debugging:
def main():
    try:
        # has been run from sys command line after conda activate
        userID = sys.argv[1]
        expID = sys.argv[2]
        print('Parameters received via command line')
    except:
        # debug mode
        print('Parameters received via debug mode')
        userID = 'adamranson'
        expID = '2025-04-13_01_ESYB007'

    #profiler = cProfile.Profile()
    #profiler.enable()

    preprocess_pupil_run(userID, expID)

    #profiler.disable()
    #profiler.print_stats(sort='time')  # or 'cumulative', 'calls', etc.
       

if __name__ == "__main__":
    main()