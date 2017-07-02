
import pickle
import numpy as np
import matplotlib.image as mpimg
import os
import glob
import cv2

############ Functions ############


# Loads dictionary from pickle
def load_data(file_name):
    if os.path.isfile(file_name):
        with open(file_name, mode='rb') as f:
            return pickle.load(f)
    return None


# Stores dictionary to pickle
def save_data(file_name, data):
    with open(file_name, mode='wb') as f:
        pickle.dump(data, f)


# Returns the camera-intrinsic correction coefficients based on calibration images in a given folder
def calibrate_camera(img_folder, check_width=9, check_height=6):
    # Retrieves normal chessboard corners from numpy
    obj_corners = np.zeros((check_width * check_height, 3), np.float32)
    obj_corners[:, :2] = np.mgrid[0:check_width, 0:check_height].T.reshape(-1, 2)

    # Collect image chessboard corners (img_points) and associated normal chessboard corners (obj_points)
    img_points = []
    obj_points = []
    images = glob.glob('{}/calibration*.jpg'.format(img_folder))
    shape = [0, 0]
    for image in images:
        img = mpimg.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if shape == [0, 0]:
            shape = gray.shape[::-1]
        elif shape != gray.shape[::-1]:
            print("Warning, unexpected shape {} in {}, rescaling image to {}..".format(gray.shape[::-1], image, shape))
            gray = cv2.resize(gray, shape)

        ret, img_corners = cv2.findChessboardCorners(gray, (check_width, check_height), None)
        if ret:
            img_points.append(img_corners)
            obj_points.append(obj_corners)
            img = cv2.drawChessboardCorners(img, (check_width, check_height), img_corners, ret)
            mpimg.imsave(image.replace('calibration', 'corners'), img, format='jpg')
        else:
            print("Error, failed to detect chessboard in {}".format(image))

    # Calculate correction coefficients based on gathered points
    ret, mtx, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, shape, None, None)

    if ret:
        return {'camera_matrix': mtx, 'dist_coefficients': dist}
    else:
        print("Error, calibration failed.")
        return None


############# Classes #############


# Represents camera calibration object, buffering calibration data in file
class CameraCalibration:

    # Construct new calibration object
    def __init__(self):
        self.cam_calibration = load_data('camera_cal/camera_cal.p')
        if self.cam_calibration is not None:
            print("Camera calibration loaded.")
        else:
            self.cam_calibration = calibrate_camera('camera_cal', 9, 6)
            if self.cam_calibration is not None:
                print("Camera calibration calculated.")
                save_data('camera_cal/camera_cal.p', self.cam_calibration)

    # Returns undistorted image
    def undistort(self, img):
        return cv2.undistort(
            img,
            self.cam_calibration['camera_matrix'],
            self.cam_calibration['dist_coefficients'],
            None,
            self.cam_calibration['camera_matrix'])
