import numpy as np
import cv2

# Sobel gradient in one direction and thresholding
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    thresh_min = thresh[0]
    thresh_max = thresh[1]

    # Convert BGR to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = abs_sobel #np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output

# Magnitude of the gradient
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

# Direction of the Gradient
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

def extractpixels(imgBGR, plotall=False, plotid=0):
    ###############################################
    # Threholding to extract pixels
    ###############################################
    # Convert to grayscale
    gray = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2GRAY)

    # Choose a Sobel kernel size
    ksize = 3 # Choose a larger odd number to smooth gradient measurements

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(imgBGR, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    grady = abs_sobel_thresh(imgBGR, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    mag_binary = mag_thresh(imgBGR, sobel_kernel=ksize, mag_thresh=(50, 80))#(30, 100))
    dir_binary = dir_threshold(imgBGR, sobel_kernel=ksize, thresh=(0.7, 1.3))

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    # Return the combined image
    return combined

#    # extract s-channel of HLS colorspace
#    hls = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HLS)
#    s_channel = hls[:,:,2]
#
#    # Threshold color channel
#    s_thresh_min = 0
#    s_thresh_max = 255
#    s_binary = np.zeros_like(s_channel)
#    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
#
#    combined2 = np.zeros_like(dir_binary)
#    combined2[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (s_binary == 1)] = 1
#
#    # Return the combined image
#    return combined

def perspectiveTransform(undst_rgb, img_bin, plotall=False, plotid=0):
    # Define perspective transformation area
    img_size = (img_bin.shape[1], img_bin.shape[0]) #(1280, 720)

	#[8,220],[200,30],[410,30],[640,220]
	#[0,290],[200,100],[440,100],[640,290]
    top_left_src = [200,100] #[578, 460]
    top_right_src = [440,100] #[702, 460]
    bottom_right_src = [640,290] #[1088, 720]
    bottom_left_src = [0,290] #[192, 720]

    top_left_dst = [200,100] #[320, 0]
    top_right_dst = [440,100] #[960, 0]
    bottom_right_dst = [440, 290] #[960, 720]
    bottom_left_dst = [200,290] #[320,720]

    src = np.float32( [ top_left_src, top_right_src, bottom_right_src, bottom_left_src ] )
    dst = np.float32( [ top_left_dst, top_right_dst, bottom_right_dst, bottom_left_dst ] )

    # perform transformation
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(undst_rgb, M, img_size, flags=cv2.INTER_LINEAR)
    warped_bin = cv2.warpPerspective(img_bin, M, img_size, flags=cv2.INTER_LINEAR)

    return M, Minv, warped, warped_bin

def rect_is_all_white(img_bin,x1,y1,x2,y2):
	if (np.amax(img_bin[y1:y2,x1:x2])>0.):
		return False
	else:
		return True

def find_white_patch(img_bin):
	nrows=480 # img_bin.shape[0] #480
	ncols=640 # img_bin.shape[1] #640
	xstep = 10
	for ss in range(nrows,10,-10):
		xtravel = int((ncols-ss)/2/xstep)
		for xx in range(0,xtravel+1):	
			xshift = xx*xstep	
			#shift left
			if rect_is_all_white(img_bin,int(ncols/2-ss/2-xshift),nrows-ss,int(ncols/2+ss/2-xshift),nrows):
				return ss, -xshift
			#shift right
			if rect_is_all_white(img_bin,int(ncols/2-ss/2+xshift),nrows-ss,int(ncols/2+ss/2+xshift),nrows):
				return ss, xshift
	return -1,0 
