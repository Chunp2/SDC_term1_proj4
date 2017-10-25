## Advanced Lane Finding

### Author: Paul Chun
### Date: OCT. 25 2017

---

**Advanced Lane Finding Project**

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

[image1]: ./images/undist_chessboard.png "Undistorted"
[image2]: ./images/undist_road.png "Road Transformed"
[image3]: ./images/unwarped_bin.png "Binary Example"
[image4]: ./images/unwarped_img.png "Warp Example"
[image5]: ./images/window.png "Fit Visual"
[image6]: ./images/image_output.png "Output"
[video1]: ./project_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
---
### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second code cell of the IPython notebook located in "./P4.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]
The position of the white car has moved a little to the right. This shows that the image has been undistorted.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of HSV, HSL, and RGB Color spaces to generate a binary image.
(The steps at lines 1 through 28 in 13th cell in 'P4.py'). Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in 12th cell in the file `P4.ipynb`.  The `unwarp()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose to hardcode the source and destination points in the following manner:

```python
src = np.float32([(0.42*width,0.68*height),
                  (0.60*width, 0.68*height),
                  (0.21*width, 0.94*height),
                  (0.85*width, 0.94*height)])
dst = np.float32([(0.30*width,0),(0.70*width,0),(0.30*width,height),(0.70*width,height)])
```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 537, 489      | 384, 0        |
| 768, 489      | 896, 0        |
| 268, 676      | 384, 720      |
| 1088, 676     | 896, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I created rectangles using cv2.rectangle() surrounding the lane lines with size of 200(width) by 80(height). The lane line pixels within the rectangles are considered to be fit to my lanes lines for both left and right, using 2nd order polynomial and it looks like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in 20th cell in my code in `P4.ipynb`
```python
def calc_curvature1 ( img, left_fit, right_fit):
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    y_eval = np.max(ploty)

    # Convert from Pixel to meter
    ym_per_px = 8/720
    xm_per_px = 3.7/700

    # Find 2nd order polynomial curve
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Poly fit in meter
    left_fit_cr = np.polyfit(ploty*ym_per_px, left_fitx*xm_per_px, 2 )
    right_fit_cr = np.polyfit(ploty*ym_per_px, right_fitx*xm_per_px, 2)

    # Radius of Curvature Calculation
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_px + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_px + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    # Calculate the lane center
    car_center = img.shape[1]/2
    left_fit_x_int = left_fit[0]*img.shape[0]**2 + left_fit[1]*img.shape[0] + left_fit[2]
    right_fit_x_int = right_fit[0]*img.shape[0]**2 + right_fit[1]*img.shape[0] + right_fit[2]
    lane_center = (left_fit_x_int +  right_fit_x_int)/2

    # Calculate the distance from the center of car in meter
    center_dist = (car_center - lane_center) * xm_per_px

    return left_curverad, right_curverad, center_dist
```
#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in 27th cell in my code in `P4.ipynb` in the function `process_image()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Initially, I tried to identify the lane lines with gradient threshold, but they didn't work well in identifying lanes under the shadow(ex ./test_images/test5.jpg). So I tried other colorspaces such as Lab, LUV, RGB, HSL, and HSV color spaces. The most challenging and time consuming part of this project was to test every color spaces to come up with the best ones that identifies the yellow lane lines under the shadow. In addition, I found it very challenging to identify yellow line at where the road color changes from dark gray to bright brown. My pipeline always failed at the transition of the road color, and it took me a while to find the best threshold that can process the transition smoothly. Currently, my pipeline is more robust than before, but there is still a few problem detecting the yellow line. I found that my pipeline draws line on the yellow line reflector instead of the yellow line in the beginning of the video. I think that this problem can be solved by masking out the inner side between the two lane lines.
