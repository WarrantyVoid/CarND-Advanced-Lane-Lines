

# Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./examples/undistort_lanes.png "Road Transformed"
[image3]: ./examples/area_of_interest.png "Transformation area"
[image4]: ./examples/warped_straight_lines.png "Warp Example"
[image5]: ./examples/color_spaces.png "Color spaces"
[image6]: ./examples/binary_combo_example.png "Binary Example"
[image7]: ./examples/color_fit_lines.png "Fit Visual"
[image8]: ./examples/example_output.png "Output"


## Rubric Points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/571/view) individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the module `calibration.py`.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `obj_corners` is just a replicated array of coordinates, and `obj_points` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `img_points` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `obj_points` and `img_points` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![Undistorted][image1]

I have saved all the images with marked `img_points` to `camera_cal/corners<num>.jpg` and all undistorted calibration images to `camera_cal/undist<num>.jpg`. The images `camera_cal/calibration1.jpg` and `camera_cal/calibration4.jpg` could not be used successfully for calibration because some of the chessboard corners were outside of the images.

I've saved the calibration results to the pickle file `camera_cal.p`, so I would not need to repeat the calibration every time I've launched the lane recognition.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![Road Transformed][image2]

Once we have obtained the camera matrix and the distortion coefficients k1, k2, p1, p2, etc. from calibration, the application of the radial and tangential distortion correction is incredibly easy. It just involves calling a single method in OpenCV which does the required matrix multiplications behind the scenes. We can visually notice in the above image example, how the radials distortion correction shifted the pixels of the white car further to the image boundary and restored the correct curve of hood on the image bottom. 

I've written the class `CameraCalibration` in `calibration.py` which encapsulates all of the calibration and un-distortion mechanics behind a simple facade.


#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is encapsulated inside the class `AreaOfInterest` in `lane_detection.py` which I harvested from the first project. The class allows to specify an area of interest polygon given four percentage values relative to image size. The area can be displayed for verification, but also used to apply a perspective transformation. 

![Transformation area][image3]

During the course of testing with different videos, I realized how sensitive the perspective transformation was about the chosen destination points. Points that were wielding an almost perfect transformation in the project video, were causing noticeably bad transformations in the challenge video due to slightly different road slope / camera mount angle. For this reason, I used slightly different parameters for the challenge video.

This resulted in the following source and destination points (1280x720):

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 560, 450      | 150, 0        | 
| 720, 450      | 1130, 0       |
| 1280, 670     | 1130, 720     |
| 0, 670        | 150, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image:

![Warp Example][image4]

I used blue as fill color during transformation as this interfered the least with my color thresholding strategy.

#### 3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of three color and gradient thresholds to generate a binary image. The code for this can be found in the file `thresholding.py`.

At first I was analyzing multiple colors spaces based on some difficult image examples, e.g.:

![Color spaces][image5]

The channel map shows, that the saturation channel of the HLS space did an overly well good job at detecting the yellow lanes, but it came with a strong downside. Dark sections and especially shadows appear "bright white" in that channel, which can result in terrible thresholding results for many occasions. As conclusion, I've decided to go with the B (blue->yellow) channel of the LAB color space instead, which has no such problem. And since I already translated to LAB, I used the L channel for detecting  white lanes. 

As thresholding technique, I split the image into five horizontal slices and performed the thresholding based on the deviation from mean value for each slice. This helped a bit for dealing with road section with different brightness values. The function name is `adaptive_threshold()`

Then, I used edge thresholding in a tricky way in function `symmetric_sobel_thresh()`. I wanted to make use of the fact, that line lines must have two outer borders each. So I used the Sobel operator into x direction onto the already perspective translated image with a very low threshold. Doing so, I retrieved the edge pixels for gradients < 0 and gradients > 0 separately. Then I shifted both resulting edges (right edges to the left, left edges to the right) so that they would overlap and applied a bitwise AND operation over the two results.

At last, for detecting the white lines, I used another AND operation on the thresholded white pixels together with the edge pixels and an AND operation on the thresholded yellow pixels together with the edge pixels. For the start screen shot in the `harder_challenge_video.mp4` the result looks quite promising:

![Binary Example][image6]


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The detection of lane pixels can be found in the file `udacity.py`. Here I assembled all of the example code from the project description into according methods. The function names are `find_lane_points()` for the initial search and `find_lane_points_inc()` for the incremental search of lane points. I did not change much of the code, as I am not proficient with Python and the scope of this project was only 2 weeks. However, I fixed one bug in the routine:

The sample code from "33. Finding Lanes, Sliding Window" finds the starting points of the sliding window based on the histogram over the lower half of the image. Instead, however, I've changed it so, that it only uses the lower 3rd.

Another small thing that I've tweaked is, that the window keeps sliding by the latest horizontal delta in case no pixels have been detected at all. This decreases the error for the polynomial match.

Another way to potentially improve the algorithm could be to not just focus on the centers of the non-zero pixels, but also on the direction of the point cloud (similar to Principal component analysis). However, I did not follow up on that approach. 

Finally, after the points belonging to the lanes have been identified, I've used the numpy library to fit a vertical polynomial to each lane.

In the pipeline, I always keep a history of 10 line matching results. Instead of return the most recent match, I instead calculate the mean match over the successful (=plausible) matches in the history. Whenever possible, I've used the incremental lane pixel search.

![Fit Visual][image7]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The calculation of lane curvature can be also found in the file `udacity.py`.  I did not change anything to the sample code. I've just put together the formula which allowed to get an approximation of the curve radius by dividing the first and second derivatives of the polynomial plus the part where the coordinate system is transferred from pixel to m using approximate conversion factors. 
 
 The calculation of the vehicle position was not given as sample code and can be found in the file `lane_detection.py` in the function `calc_car_offset()`. The implementation is trivial though.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I've ported the given project sample code in function `draw_lane_area()` inside `udacity.py`. Here is an example of my result on a test image:

![Output][image8]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

The pipeline performs perfectly on the first two videos, but delivers awful results for the last video:

* [project_video.mp4](./output_videos/project_video.mp4)
* [challenge_video.mp4](./output_videos/challenge_video.mp4)
* [harder_challenge_video.mp4](./output_videos/harder_challenge_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipeline uses the function `is_plausible()` in order to determine, whether the currently found lane polynomials..
* ..are parallel enough (by comparing min and max distances in between lane lines)
* ..span a lane area of plausible width (by comparing mean distance in between lane lines)

However, already a small error in perspective transformation can cause the lane lines to become much less parallel and as thus being rejected by the plausibility check. Maybe in an advanced system there could be a back pass that is constantly adapting and correcting perspective transformation based on how much the lane lines are parallel.

Another weak point in the pipeline is the sliding window approach to find lane pixels. It only tracks down one possible path for each lane following purely the centers of the pixel blocks. This makes it vulnerable against outliers and once the window made an incorrect slide, it will keep following the false track. A better approach might expand a tree of possibilities (similar to A*) and only decide at the very end which path is the most likely one. 

Other than the obvious problems with strong reflections, extreme lighting conditions and very tight curves, the pipeline has proven quite powerful :)
