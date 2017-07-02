
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
import calibration
import thresholding
import udacity


############ Functions ############


# Calculates car offset from lane center
def calc_car_offset(image_size, lane_lines):
    left_fit = lane_lines[0]
    right_fit = lane_lines[1]

    y = image_size[0]
    left_fitx = left_fit[0] * y ** 2 + left_fit[1] * y + left_fit[2]
    right_fitx = right_fit[0] * y ** 2 + right_fit[1] * y + right_fit[2]
    offset_x = (left_fitx + right_fitx - image_size[1]) / 2 * udacity.X_METER_PER_PIXEL
    return offset_x


# Draw information overlay containing curve and offset
def draw_info(img, curve_rad, car_offset):
    rad_text = "{}: {:.2f}m".format('Curve', (curve_rad[0] + curve_rad[1]) / 2)
    off_text = "{}: {:.2f}m".format('Offset', car_offset)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, rad_text, (20, 70), font, 2, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.putText(img, off_text, (20, 140), font, 2, (0, 255, 0), 3, cv2.LINE_AA)
    return img


# Checks whether the calculates lane lines are plausible
def is_plausible(img, lane_lines):
    if lane_lines is None:
        return False
    left_fit = lane_lines[0]
    right_fit = lane_lines[1]
    ploty = np.linspace(0, img.shape[0] - 1, num=img.shape[0])  # to cover same y-range as image
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    delta_fitx = right_fitx - left_fitx
    delta_mean = np.mean(delta_fitx)
    delta_min = np.min(delta_fitx)
    delta_max = np.max(delta_fitx)
    if delta_mean > 3.7 / udacity.X_METER_PER_PIXEL * 1.1:
        return False
    if delta_mean < 2.0 / udacity.X_METER_PER_PIXEL * 0.9:
        return False
    if delta_min < delta_mean * 0.75:
        return False
    if delta_max > delta_mean * 1.25:
        return False
    return True


############# Classes #############


# Represents an area of interest inside an image, implements perspective transform
class AreaOfInterest:

    # Constructs an area of interest given an image size and rectangle parameters in percent
    def __init__(self, image_size, width1_p, width2_p, height1_p, height2_p):
        self.image_size = image_size
        h = self.image_size[0]
        w = self.image_size[1]
        self.poly = np.float32([
            [w * 0.5 * (1 - width2_p), h * (1 - height2_p)],
            [w * 0.5 * (1 + width2_p), h * (1 - height2_p)],
            [w * 0.5 * (1 + width1_p), h * (1 - height1_p)],
            [w * 0.5 * (1 - width1_p), h * (1 - height1_p)]])
        offsetx = 150
        offsety = 0
        self.dst = np.float32([
            [offsetx, offsety],
            [image_size[1] - offsetx, offsety],
            [image_size[1] - offsetx, image_size[0] - offsety],
            [offsetx, image_size[0] - offsety]])
        self.M = cv2.getPerspectiveTransform(self.poly, self.dst)
        self.M_inv = cv2.getPerspectiveTransform(self.dst, self.poly)

    # Draws the area into an image
    def draw(self, image, color=(255, 255, 255), width=3):
        cv2.polylines(image, [self.poly.astype(np.int32)], isClosed=True, color=color, thickness=width)

    # Translates the area into image
    def translate(self, img):
        return cv2.warpPerspective(img, self.M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 255))


# Represents a lane line processor for images
class ImageProcessor:
    # Constructs the processor with given image/frame size
    def __init__(self, camera_calibration, image_size, corr_p):
        self.camera_calibration = camera_calibration
        self.area_of_interest = AreaOfInterest(image_size, 1.0, 0.125, 0.07, corr_p)
        self.image_size = image_size
        self.lane_line_history = []
        self.lane_lines = None
        self.y_axis = np.linspace(0, image_size[0] - 1, num=image_size[0]).astype(np.int32)

    # Retrieves x values from lane line polynomials
    def get_lane_lines_x(self, lane_lines):
        if lane_lines is not None:
            lf = lane_lines[0]
            rf = lane_lines[1]
            y = self.y_axis
            left_x = lf[0] * y ** 2 + lf[1] * y + lf[2]
            right_x = rf[0] * y ** 2 + rf[1] * y + rf[2]
            return left_x, right_x
        return None

    # Adds low-pass filter to final lane lines
    def filter_lanes(self, img, lane_lines, window_size=10):
        # Append x values
        self.lane_line_history.append(self.get_lane_lines_x(lane_lines))

        # Enforce history size
        if len(self.lane_line_history) > window_size:
            self.lane_line_history.pop(0)

        # Calculate mean based on history
        x_mean_left = np.zeros(img.shape[0])
        x_mean_right = np.zeros(img.shape[0])
        x_mean_count = 0
        for lane_lines in self.lane_line_history:
            if lane_lines is not None:
                x_mean_left += lane_lines[0]
                x_mean_right += lane_lines[1]
                x_mean_count += 1
        if x_mean_count > 0:
            # Match new poly to mean
            x_mean_left = (x_mean_left / x_mean_count).astype(np.int32)
            x_mean_right = (x_mean_right / x_mean_count).astype(np.int32)
            lane_lines = udacity.match_poly(img, (x_mean_left, self.y_axis, x_mean_right, self.y_axis))
            if is_plausible(img, lane_lines):
                # Return mean
                return lane_lines
            else:
                # Corrupted history, restart over
                self.lane_line_history.clear()
                return None
        else:
            # Nothing usable in history
            return None

    # Implements the lane marking pipeline
    def pipeline(self, img):
        # Require defined image size
        assert img.shape == self.image_size

        # Undistort
        img = cam_calibration.undistort(img)

        # Perspective mapping
        top_down = self.area_of_interest.translate(img)

        # Edge/Color thresholding
        top_down = thresholding.multi_threshold(top_down)

        # Lane point identification
        if self.lane_lines is not None:
            # Incremental approach
            lane_points = udacity.find_lane_points_inc(top_down, self.lane_lines)
        else:
            # From scratch
            lane_points = udacity.find_lane_points(top_down)

        # Calc lane lane lines as polynomials
        self.lane_lines = udacity.match_poly(top_down, lane_points)

        # Check plausibility and push to history filter
        if is_plausible(top_down, self.lane_lines):
            self.lane_lines = self.filter_lanes(top_down, self.lane_lines)
        else:
            self.lane_lines = self.filter_lanes(top_down, None)

        if self.lane_lines is not None:
            # Calculate stats from lane poly lines
            curve_rad = udacity.calc_curve_radius(self.y_axis, self.lane_lines)
            car_offset = calc_car_offset(self.image_size, self.lane_lines)

            # Draw all results into image
            img = udacity.draw_lane_area(img, self.lane_lines, self.area_of_interest.M_inv)
            img = draw_info(img, curve_rad, car_offset)
        return img


############ Main logic ###########


if __name__ == '__main__':
    cam_calibration = calibration.CameraCalibration()


    # Pipeline for single picture
    rgb = mpimg.imread('test_images/test6.jpg')
    processor = ImageProcessor(cam_calibration, rgb.shape, 0.375)
    rgb = processor.pipeline(rgb)
    plt.imshow(rgb)
    mpimg.imsave('output_images/test6.jpg', rgb)
    plt.show()

    '''
    # Pipeline for video
    clip = VideoFileClip('test_videos/harder_challenge_video.mp4')
    processor = ImageProcessor(cam_calibration, (clip.h, clip.w, 3), 0.35)
    new_clip = clip.fl_image(lambda frame: processor.pipeline(frame))
    new_clip_output = 'output_videos/harder_challenge_video.mp4'
    new_clip.write_videofile(new_clip_output, audio=False)
    '''
