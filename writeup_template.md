[//]: # (Image References)

[image1]: ./output_images/undistort_cal.jpg "Undistorted"
[image2]: ./output_images/undistort.jpg "Road Transformed"
[image3]: ./output_images/binary.jpg "Binary Example"
[image4]: ./output_images/wrap.jpg "Warp Example"
[image5]: ./output_images/sliding_window.jpg "Fit Visual"
[image6]: ./output_images/output.jpg "Output"

###Writeup / README

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The images are undistorted using the ImageUndistortor class (lines 7-53 in process_video.py). To calibrate it, I transform the image to grayscale and then apply cv2.findChessboardCorners on the provided images.
After that I use cv2.calibrateCamera to get the calibration matrix.
When the ImageUndistortor class is ready, I use its undistort method for processing images. It is implemented using cv2.undistort
![alt text][image1]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
![alt text][image2]
####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I use various transformations:
- Sobel gradient in x/y direction
- Magnitude of Sobel gradient
- Direction of Sobel gradient
- Using only the S channel from the HLS colorspace

Those are defined in the ImageThresholder class (line 113)
All this is combined to receive the final result.


![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The perspective transformation is applied using the class PerspectiveTransformator(line 56).
It calculates the transform and reverse transform matricies on initializaiton
After that I use the transform method to transform images.
The source and destination points are predifined in the class:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 277, 670      | 200, 720      | 
| 581, 460      | 200, 0        |
| 701, 460      | 980, 0        |
| 1028, 670     | 980, 720      |



![alt text][image4]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The pixels are indentified by using a sliding window algorithm.
The first step is to find where to start looking for lane lines. This is done in two ways:
- If there is no previous information, we calculate a histogram and use the peaks for starting points
- If there is information from previous frames, we use this to estimate where the lines should be in this frame. They change only a few pixels from frame to frame
After we have the starting points, we start searching for the areas with the most points and move our window to "catch" as many as possible in its width.
This is done in the LineDetector class (line 246)

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

This is done in the calc_curvative method of the VideoLineDrawer class (line 476) and it is almost identical to the example code.
We fit a second degree polynomial to the points (before the reverse perspective transform) and calculate the curve radius using the provided formula
The position of the vehicle is calculated by calculating the position of the left and right lane line at the bottom of the image, finding the center between them and comparing it to the center of the image 

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.


![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.out.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

There are a lot of parameters to tweak (e.g. the thresholds in ImageThresholder). This makes development very slow and painful. 
Also, this parameters work only for this video and would likely break in different conditions.
Making it more robust would involve some way of calculating and adjusting them dynamically.
Also, having a few different ways for finding line markings would help in situations where one of the methods fails.

