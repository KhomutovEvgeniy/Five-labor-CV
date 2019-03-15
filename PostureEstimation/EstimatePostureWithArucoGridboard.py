# The following code is used to watch a video stream, detect Aruco markers, and use
# a set of markers to determine the posture of the camera in relation to the plane
# of markers.
#
# Assumes that all markers are on the same plane, for example on the same piece of paper
#
# Requires camera calibration (see the rest of the project for example calibration)

import numpy
import cv2
import cv2.aruco as aruco
import os
import pickle


def draw(img, corners, rvec, tvec, cameraMatrix, distCoeffs):
    axis = numpy.float32([[0, 0, 0], [0, 3, 0], [3, 3, 0], [3, 0, 0],
                         [0, 0, -3], [0, 3, -3], [3, 3, -3], [3, 0, -3]])

    # project 3D points to image plane
    imgpts, jac = cv2.projectPoints(axis, rvec, tvec, cameraMatrix, distCoeffs)
    # print(imgpts)
    imgpts = numpy.int32(imgpts).reshape(-1, 2)
    print('imgpts', imgpts)

    # draw ground floor in green
    # img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)
    # draw pillars in blue color
    # for i, j in zip(range(4), range(4, 8)):
    #     img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)
    #
    # # draw top layer in red color
    # img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)
    # cv2.imshow('result', img)

    return img


# Check for camera calibration data
if not os.path.exists('./calibration.pckl'):
    print("You need to calibrate the camera you'll be using. See calibration project directory for details.")
    exit()
else:
    f = open('calibration.pckl', 'rb')
    (cameraMatrix, distCoeffs) = pickle.load(f)
    f.close()
    if cameraMatrix is None or distCoeffs is None:
        print("Calibration issue. Remove ./calibration.pckl and recalibrate your camera with CalibrateCamera.py.")
        exit()

# Constant parameters used in Aruco methods
ARUCO_PARAMETERS = aruco.DetectorParameters_create()
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_5X5_50)

# Create grid board object we're using in our stream
board = aruco.GridBoard_create(
        markersX=5,
        markersY=7,
        markerLength=0.04,
        markerSeparation=0.01,
        dictionary=ARUCO_DICT)

# Create vectors we'll be using for rotations and translations for postures
rvecs, tvecs = None, None

cam = cv2.VideoCapture(0)

while(cam.isOpened()):
    # Capturing each frame of our video stream
    ret, QueryImg = cam.read()
    if ret == True:
        # grayscale image
        gray = cv2.cvtColor(QueryImg, cv2.COLOR_BGR2GRAY)

        # Detect Aruco markers
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)

        # Refine detected markers
        # Eliminates markers not part of our board, adds missing markers to the board
        corners, ids, rejectedImgPoints, recoveredIds = aruco.refineDetectedMarkers(
                image = gray,
                board = board,
                detectedCorners = corners,
                detectedIds = ids,
                rejectedCorners = rejectedImgPoints,
                cameraMatrix = cameraMatrix,
                distCoeffs = distCoeffs)

        ###########################################################################
        # TODO: Add validation here to reject IDs/corners not part of a gridboard #
        ###########################################################################

        # Outline all of the markers detected in our image
        QueryImg = aruco.drawDetectedMarkers(QueryImg, corners, borderColor=(0, 0, 255))

        # Require 15 markers before drawing axis
        if ids is not None and len(ids) > 15:
            # Estimate the posture of the gridboard, which is a construction of 3D space based on the 2D video
            pose, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, cameraMatrix, distCoeffs)
            # if pose:
                # Draw the camera posture calculated from the gridboard
                QueryImg = aruco.drawAxis(QueryImg, cameraMatrix, distCoeffs, rvec, tvec, 0.3)

        if ids is not None and len(ids) > 0:
            pose, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, cameraMatrix, distCoeffs)

            cube = draw(QueryImg, corners, rvec, tvec, cameraMatrix, distCoeffs)
            cv2.imshow('cube', cube)
            # cv2.waitKey(0)
        # Display our image
        cv2.imshow('QueryImage', QueryImg)
    # Exit at the end of the video on the 'q' keypress
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
