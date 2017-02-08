import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

class ImageUndistortor:
    """
    This class handles undistortion of images

    It should be calibrated using the calibrate method before using it
    """

    def calibrate(self, images_filename_pattern="camera_cal/calibration*.jpg"):
        """
        Calibrates the ImageUndistortor using pictures of a chessboard 

        Parameters
        ----------
        images_filename_pattern : string
            File location for the calibration images
        """
        objp = np.zeros((9*6,3), np.float32)
        objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

        objpoints = []
        imgpoints = []
        
        cal_images = glob.glob(images_filename_pattern)
        for image_path in cal_images:
            image = mpimg.imread(image_path)
            gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
            found, corners = cv2.findChessboardCorners(gray, (9,6),None)
            if found:
                imgpoints.append(corners)
                objpoints.append(objp)
                
        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

    def undistort(self, image):
        """
        Undistorts a single image with already calculated matrix

        Parameters
        ----------
        image : numpy array
            The image to undistort
        Returns
        -------
        image : numpy array
            The undistorted image
        """
        return cv2.undistort(image, self.mtx, self.dist, None, self.mtx)


class PerspectiveTransformator:
    """
    This class handles perspective transformation of images

    It uses a predefined set of points for calculating the undistort matrix
    """
    
    def __init__(self):
        """
        Calculates transform and reverse transform matrix for perspective transformation
        """
        src_coords = np.float32([
            [277, 670],
            [581, 460],
            [701, 460],
            [1028, 670]
        ])
        dst_coords = np.float32([
            [200, 720],
            [200, 0],
            [980, 0],
            [980, 720]
        ])
        self.transformMatrix = cv2.getPerspectiveTransform(src_coords, dst_coords)
        self.reverseTransformMatrix = cv2.getPerspectiveTransform(dst_coords, src_coords)
        
    def transform(self, image):
        """
        Applies the precomputed transformation on an image

        Parameters
        ----------
        image : numpy array
            The image to transform
        Returns
        -------
        image : numpy array
            The transformed image
        """
        img_size = (image.shape[1], image.shape[0])
        return cv2.warpPerspective(image, self.transformMatrix, img_size, flags=cv2.INTER_LINEAR)
        
    def reverse_transform_points(self, points):
        """
        Applies reverse transformation on a set of points

        Parameters
        ----------
        points : numpy array
            The points on which to apply the transformation
        Returns
        -------
        points : numpy array
            The transformed points
        """
        return cv2.perspectiveTransform(points, self.reverseTransformMatrix)

class ImageThresholder:
    """
    This class contains several static methods for transforming a source image to binary output
    """
    
    @staticmethod
    def abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(0, 255)):
        """
        Calculates directional gradient & applies threshold

        Parameters
        ----------
        image : numpy array
            The image to process
        Returns
        -------
        image : numpy array
            The binary output image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if orient == 'x':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        else:
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
            
        absolute = np.absolute(sobel)
        scaled = np.uint8(absolute*255/np.max(absolute))
        binary_output = np.zeros_like(scaled)
        binary_output[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1
        
        return binary_output
        
    @staticmethod
    def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
        """
        Calculates gradient magnitude & applies threshold

        Parameters
        ----------
        image : numpy array
            The image to process
        Returns
        -------
        image : numpy array
            The binary output image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        mag = np.sqrt(sobelx ** 2 + sobely ** 2)
        scaled = np.uint8(mag*255/np.max(mag))
        
        binary_output = np.zeros_like(scaled)
        binary_output[(scaled >= mag_thresh[0]) & (scaled <= mag_thresh[1])] = 1
        
        return binary_output
    
    @staticmethod
    def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
        """
        Calculates gradient direction & applies threshold

        Parameters
        ----------
        image : numpy array
            The image to process
        Returns
        -------
        image : numpy array
            The binary output image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
        sobely = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
        direction = np.arctan2(sobely, sobelx)

        binary_output = np.zeros_like(direction)
        binary_output[(direction >= thresh[0]) & (direction <= thresh[1])] = 1
        
        return binary_output
    
    @staticmethod
    def hls_threshold(image, thresh=(170, 255)):
        """
        Transform the image to HLS space, gets the S channel & applies threshold

        Parameters
        ----------
        image : numpy array
            The image to process
        Returns
        -------
        image : numpy array
            The binary output image
        """
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        s_only = hls[:, :, 2]
        
        binary_output = np.zeros_like(s_only)
        binary_output[(s_only >= thresh[0]) & (s_only <= thresh[1])] = 1
        
        return binary_output
        
    @staticmethod
    def combined(image):
        """
        Applies varies transformation on an image and combines them in one binary output

        Parameters
        ----------
        image : numpy array
            The image to process
        Returns
        -------
        image : numpy array
            The binary output image
        """
        ksize = 5
        gradx = ImageThresholder.abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(50, 200))
        grady = ImageThresholder.abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(50, 200))
        mag_binary = ImageThresholder.mag_thresh(image, sobel_kernel=ksize, mag_thresh=(10, 80))
        dir_binary = ImageThresholder.dir_threshold(image, sobel_kernel=ksize, thresh=(0.0, 0.3))
        
        combined = np.zeros_like(dir_binary)
        combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
        
        hls_binary = ImageThresholder.hls_threshold(image, thresh=(200, 255))
        
        combined_hls = np.zeros_like(hls_binary)
        combined_hls[(hls_binary == 1) | (combined == 1)] = 1
        
        return combined_hls, combined, hls_binary
    
class LineDetector:
    x_size = 30
    y_step = 100
    
    def sliding_window_step(self, image, start_x, end_y, x_search_region = 100):
        arr = np.empty((2 * x_search_region))
        
        start_y = np.max((end_y - self.y_step, 0))
        for i in range(- x_search_region, x_search_region):
            x_start = int(start_x + i - self.x_size)
            x_end = int(start_x + i + self.x_size)
            arr[x_search_region + i] = np.sum(image[start_y:end_y, x_start:x_end])
            
        if np.argmax(arr) < 20:
            next_step_x = start_x
        else:
            next_step_x = np.argmax(arr) - x_search_region + start_x
        
        return next_step_x, start_y
        
    def sliding_window_one_side(self, image, start_x, output, func=None):
        indicies_x = []
        indicies_y = []
        
        current_step_x = start_x
        current_step_y = image.shape[0]
        
        cv2.rectangle(output, (int(start_x - self.x_size), current_step_y), 
                              (int(start_x + self.x_size), current_step_y - self.y_step), 1, thickness=15)
        
        while current_step_y > 0:
            if func != None:
                current_step_x = func(current_step_y)
                next_step_x, next_step_y = self.sliding_window_step(image, current_step_x, current_step_y, 50)
            else:
                next_step_x, next_step_y = self.sliding_window_step(image, current_step_x, current_step_y)
            
            arr = image[next_step_y: current_step_y, next_step_x - self.x_size: next_step_x + self.x_size]
            current_indicies = np.where( arr == 1)
            indicies_y.append(current_indicies[0] +  next_step_y)
            indicies_x.append(current_indicies[1] +  (next_step_x - self.x_size))
            
            cv2.rectangle(output, (int(next_step_x - self.x_size), current_step_y), 
                                  (int(next_step_x + self.x_size), next_step_y), 1, thickness=5)
            current_step_x = next_step_x
            current_step_y = next_step_y

        return np.array((np.concatenate(indicies_x), np.concatenate(indicies_y)), dtype='float64')
    
    def sliding_window(self, image, l_points_prev=None, r_points_prev=None, x_region=50):
        output = np.copy(image)
        
        
        if l_points_prev == None or r_points_prev == None:            
            start_left, start_right = self.get_starting_points_histogram(image)
            left_indicies = self.sliding_window_one_side(image, start_left, output)
            right_indicies = self.sliding_window_one_side(image, start_right, output)
        else:
            left_func = self.get_starting_points_previous(l_points_prev)
            right_func = self.get_starting_points_previous(r_points_prev)
            start_left = left_func(720)
            start_right = right_func(720)
            
            left_indicies = self.sliding_window_one_side(image, start_left, output, left_func)
            right_indicies = self.sliding_window_one_side(image, start_right, output, right_func)

        
        return left_indicies, right_indicies, output
    
    def get_starting_points_previous(self, points):
        polyfit = np.polyfit(points[1], points[0], 2)
        return lambda y: polyfit[0]*y**2 + polyfit[1]*y + polyfit[2]
        
    def get_starting_points_histogram(self, image, x_region=50):
        histogram = np.sum(image[image.shape[0]/2:,:], axis=0)
        sliding_peaks = np.empty((histogram.shape[0]))

        for i in range(0, histogram.shape[0]):
            sliding_peaks[i] = np.sum(histogram[i:(i + 2 * x_region)])
        
        start_left = np.argmax(sliding_peaks[0:sliding_peaks.shape[0]/2])
        start_right = np.argmax(sliding_peaks[sliding_peaks.shape[0]/2:-1]) + sliding_peaks.shape[0]/2
        return start_left, start_right
        
class VideoLineDrawer:
    imageUndistortor = ImageUndistortor()    
    perspectiveTransformator = PerspectiveTransformator()
    lineDetector = LineDetector()
        

    left_fit_prev = None
    right_fit_prev = None
    left_curverad_prev = None
    right_curverad_prev = None
    l_points = None
    r_points = None

    ym_per_pix = 3/110 # meters per pixel in y dimension
    xm_per_pix = 3.7/780 # meters per pixel in x dimension

    def __init__(self):
        self.imageUndistortor.calibrate()
    
    def get_fitting_function(self, polyfit):
        return lambda y: polyfit[0]*y**2 + polyfit[1]*y + polyfit[2]
    
    def calc_curvative(self, l_points, r_points):
        l_points = np.array(l_points)
        r_points = np.array(r_points)
        
        left_fit = np.polyfit(l_points[1] * self.ym_per_pix, l_points[0] * self.xm_per_pix, 2)
        right_fit = np.polyfit(r_points[1] * self.ym_per_pix, r_points[0] * self.xm_per_pix, 2)
        
        y_eval = 720 * self.ym_per_pix
        left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
        right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
        
        y_eval_distance = 720 * self.ym_per_pix
        left_x = left_fit[0]*y_eval_distance**2 + left_fit[1]*y_eval_distance + left_fit[2]
        right_x = right_fit[0]*y_eval_distance**2 + right_fit[1]*y_eval_distance + right_fit[2]
        
        return left_curverad, right_curverad, left_x, right_x
    
    def transform_array(self, input_points):
        transposed = np.array(input_points).T
        points = transposed.reshape(1, transposed.shape[0], -1)
        return self.perspectiveTransformator.reverse_transform_points(points)[0].T
        
    def draw_line_markings(self, image, estimated, max_y_marking=450):
        y_distance = 720 - max_y_marking
        
        pts = np.empty((y_distance * 2, 2), np.int32)
        
        left_func = self.get_fitting_function(self.left_fit_prev)
        for i in range(0, y_distance, 1):
            y_coord = max_y_marking + i;
            pts[i][0] = left_func(y_coord)
            pts[i][1] = y_coord;
            
        right_func = self.get_fitting_function(self.right_fit_prev)
        for i in range(0, y_distance, 1):
            y_coord = max_y_marking + i
            index = y_distance * 2 - i - 1
            pts[index][0] = right_func(y_coord)
            pts[index][1] = y_coord

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(image).astype(np.uint8)


        # Draw the lane onto the warped blank image
        color = (0,255, 0)
        if estimated:
            color = (255,0, 0)
        cv2.fillPoly(warp_zero, [pts], color)
        # Combine the result with the original image
        return cv2.addWeighted(image, 1, warp_zero, 0.3, 0)
        
    def get_points(self, image, output):
        undistorted = self.imageUndistortor.undistort(image)
        warped = self.perspectiveTransformator.transform(undistorted)
        binary, combined, hls_binary = ImageThresholder.combined(warped)
        self.l_points, self.r_points, output_sliding = self.lineDetector.sliding_window(binary, self.l_points, self.r_points)

        
        output[720:1440, 0:1280, :] = warped
        
        output[0:720, 1280:2560, 0] = binary*255
        output[0:720, 1280:2560, 1] = binary*255
        output[0:720, 1280:2560, 2] = binary*255
        
        output[720:1440, 1280:2560, 0] = output_sliding*255
        output[720:1440, 1280:2560, 1] = output_sliding*255
        output[720:1440, 1280:2560, 2] = output_sliding*255
        
        output[0:720, 2560:3840, 0] = combined*255
        output[0:720, 2560:3840, 1] = combined*255
        output[0:720, 2560:3840, 2] = combined*255
        
        output[720:1440, 2560:3840, 0] = hls_binary*255
        output[720:1440, 2560:3840, 1] = hls_binary*255
        output[720:1440, 2560:3840, 2] = hls_binary*255
        
    def plot_image(self, image):
        output = np.empty((1440, 3840, 3), dtype='uint8')
        
        self.get_points(image, output)
        left_curverad, right_curverad, left_x, right_x = self.calc_curvative(self.l_points, self.r_points)  

        l = self.transform_array(self.l_points)
        r = self.transform_array(self.r_points)
        
        left_fit = np.polyfit(l[1], l[0], 2)
        right_fit = np.polyfit(r[1], r[0], 2)

        line_width = right_x-left_x
        line_offset = 640*self.xm_per_pix - (line_width/2 + left_x)
     
        estimated = True

        if self.left_fit_prev != None and self.right_fit_prev != None:
            if line_width > 3.6 and line_width < 4.0:
                self.left_fit_prev =  0.7 * self.left_fit_prev  + 0.3 * left_fit
                self.right_fit_prev = 0.7 * self.right_fit_prev + 0.3 * right_fit 
                self.left_curverad_prev =  0.7 * self.left_curverad_prev  + 0.3 * left_curverad
                self.right_curverad_prev = 0.7 * self.right_curverad_prev + 0.3 * right_curverad
                estimated = False
        else:
            self.left_fit_prev = left_fit
            self.right_fit_prev = right_fit
            self.left_curverad_prev = left_curverad
            self.right_curverad_prev = right_curverad
            estimated = False

        result = self.draw_line_markings(image, estimated);

        
        cv2.putText(result,"car offset:{0:.2f} m".format(line_offset), (800,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
        cv2.putText(result,"left curve rad:{0:.2f} m".format(self.left_curverad_prev), (800,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
        cv2.putText(result,"right curve rad:{0:.2f} m".format(self.right_curverad_prev), (800,130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
        cv2.putText(result,"line width:{0:.2f} m".format(line_width), (800,160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
        
        output[0:720, 0:1280, :] = result

        return output
        
