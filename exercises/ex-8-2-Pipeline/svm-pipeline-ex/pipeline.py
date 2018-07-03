import pickle
import  matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import glob
from clf_svm import SVMCLF

###########################################################3
clf_svm = SVMCLF('model_svm.p','svm_params.p',nhistory=1)
#testmode = 'images'
#pathinput = '../testdata/frame-test/*.png'
testmode = 'video'
pathinput = '../testdata/test01.avi'

############ MAIN FUNCTION ##############

if testmode == 'images':
    for filename in glob.glob(pathinput):
        im=Image.open(filename)
        img = np.array(im)
        final_decision, draw_img = clf_svm.processOneFrame(img)

        Image.fromarray(draw_img).show()
        #print final_decision

else:
    def process(img):
        final_decision, draw_img = clf_svm.processOneFrame(img)
        return draw_img

    from moviepy.editor import VideoFileClip
    result_output = 'result_test_video_frames' + '.mp4'
    clip1 = VideoFileClip(pathinput)
    white_clip = clip1.fl_image(process)
    white_clip.write_videofile(result_output, audio=False)
