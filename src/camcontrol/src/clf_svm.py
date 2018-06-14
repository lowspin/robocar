import numpy as np
import cv2
from sklearn import svm
from scipy.ndimage.measurements import label
import imagefunctions
from tracker import Tracker
import pickle
#import random

###########################################################3
# sliding window parameters
imgsize = (128,96)
windowsize = (50,50)
slidestep = (5,5) # number of pixels to slide window
min_red_pixels = 20 # min red pixel to process window

class SVMCLF(object):
    def __init__(self, fn_model, fn_params, nhistory=1):
        self.clf = pickle.load(open(fn_model, 'rb'))
        svmparams = pickle.load(open(fn_params, 'rb')) #pickle.load(f2)
        self.fmean = svmparams['fmean']
        self.fstd = svmparams['fstd']
        self.tracker = Tracker(nhistory)

    def getFeatures(self,img):
        return [
            imagefunctions.num_corners(img),
            imagefunctions.num_edges(img),
            imagefunctions.num_red_pixels(img),
            imagefunctions.num_white_pixels(img),
            imagefunctions.abs_sobel_thresh(img, orient='y', sobel_kernel=3, thresh=(100, 200)),
            imagefunctions.mag_thresh(img, sobel_kernel=5, mag_thresh=(100, 180)),
            imagefunctions.dir_threshold(img, sobel_kernel=3, thresh=(np.pi/8, np.pi/4))
        ]

    def normalize_features(self,feature_vector,fmn,fsd):
        numDim = len(feature_vector)
        normFeatures = []
        normfeat = [None]*numDim
        for i in range(numDim):
            normfeat[i] = (feature_vector[i]-fmn[i])/fsd[i]
        normFeatures.append(normfeat)
        #transpose result
        res = np.array(normFeatures).T
        return res

    def search_windows(self,img, windows,framenum = 0):
        stop_windows=[] # list of positive stop sign detection windows
        warn_windows=[] # list of positive warn sign detection windows

        cropnum = 0
        for window in windows:
            # extract test window from orginal image
            #test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (50,50))
            img_crop = img[window[0][1]:window[1][1], window[0][0]:window[1][0]]
            #img_crop_pp = imagefunctions.preprocess_one_rgb(img_crop)
            test_img = imagefunctions.preprocess_one_rgb(img_crop)
            #test_img = np.array(255*img_crop_pp, dtype=np.uint8)

            #fname = 'crop-'+ str(framenum)+'-'+str(cropnum)+'.png'
            #imsave('img50x50/'+fname, test_img)
            cropnum = cropnum + 1
            # extract features
            feat = self.getFeatures(test_img)
            # normalize features
            normfeat = self.normalize_features(feat,self.fmean,self.fstd)
            # predict using classifier
            testvec = np.asarray(normfeat).reshape(1,-1)
            prediction = self.clf.predict(testvec)
            #print prediction
            # save positive detection windows
            if prediction == 2:
                #print 'warning sign'
                warn_windows.append(window)
            elif prediction == 1:
                stop_windows.append(window)

        # return positve detection windows
        return stop_windows, warn_windows

    def draw_labeled_bboxes(self,img, labels, boxcolor):
        # Iterate through all detected cars
        for item_number in range(1, labels[1]+1):
            # Find pixels with each item_number label value
            nonzero = (labels[0] == item_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], boxcolor, 2)
        # Return the image
        return img

    def find_signs(self,img):
        startx = 0 #60
        stopx = imgsize[0]-windowsize[0] #80
        starty = 0 #20 #19
        stopy =imgsize[1]-windowsize[1] #30

        window_list = []
        for x in range(startx, stopx, slidestep[0]):
            for y in range(starty, stopy, slidestep[1]):
                img_crop = img[ y:y+windowsize[1], x:x+windowsize[0]]
                img_crop_pp = imagefunctions.preprocess_one_rgb(img_crop)
                img_in = np.array(255*img_crop_pp, dtype=np.uint8)
                if (imagefunctions.num_red_pixels(img_in)>min_red_pixels):
                    window_list.append(((x, y), (x+windowsize[0], y+windowsize[1])))

        #stop_windows, warn_windows = self.search_windows(img, window_list, framenum=random.randint(0,9999))
        stop_windows, warn_windows = self.search_windows(img, window_list)

        # heatmap
        heat_stop = np.zeros_like(img[:,:,0]).astype(np.float)
        heat_warn = np.zeros_like(img[:,:,0]).astype(np.float)
        for bbox in window_list:
            startx = bbox[0][0]
            starty = bbox[0][1]
            endx = bbox[1][0]
            endy = bbox[1][1]
            #cv2.rectangle(img,(startx, starty),(endx, endy),(0,0,200),1)
        for bbox in warn_windows:
            startx = bbox[0][0]
            starty = bbox[0][1]
            endx = bbox[1][0]
            endy = bbox[1][1]
            heat_warn[starty:endy, startx:endx] += 1.
            #cv2.rectangle(img,(startx, starty),(endx, endy),(0,255,0),1)
        for bbox in stop_windows:
            startx = bbox[0][0]
            starty = bbox[0][1]
            endx = bbox[1][0]
            endy = bbox[1][1]
            heat_stop[starty:endy, startx:endx] += 1.
            #cv2.rectangle(img,(startx, starty),(endx, endy),(255,0,0),1)

        score_stop = np.max(heat_stop)
        score_warn = np.max(heat_warn)
        #print '[scores] stop:' + str(score_stop) + ' warn:' + str(score_warn)

        detthresh = 20
        mapthresh = 10
        labels=[None]
        if score_stop<detthresh and score_warn<detthresh:
            #print 'NO SIGN'
            decision = 0
            draw_img = img
        elif score_stop>score_warn:
            #print 'STOP'
            decision = 1
            heatmap_stop = heat_stop
            heatmap_stop[heatmap_stop <= mapthresh] = 0
            labels = label(heatmap_stop)
            #draw_img = draw_labeled_bboxes(np.copy(img), labels_stop, boxcolor=(255,0,0))
        else:
            #print 'WARNING'
            decision = 2
            # draw box
            heatmap_warn = heat_warn
            heatmap_warn[heatmap_warn <= mapthresh] = 0
            labels = label(heatmap_warn)
            #draw_img = draw_labeled_bboxes(np.copy(img), labels_warn, boxcolor=(0,255,0))

        #Image.fromarray(draw_img).show()
        return decision, labels #draw_img

    def processOneFrame(self,img):
        dec, labels = self.find_signs(img)
        self.tracker.new_data(dec)
        final_decision = self.tracker.combined_results()
        #print dec, final_decision
        draw_img = img
        if len(labels)==2:
            if final_decision==1:
                draw_img = self.draw_labeled_bboxes(np.copy(img), labels, boxcolor=(255,0,0))
            elif final_decision==2:
                draw_img = self.draw_labeled_bboxes(np.copy(img), labels, boxcolor=(0,255,0))
            else:
                draw_img = img
        return final_decision, draw_img
