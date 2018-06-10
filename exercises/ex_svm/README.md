# Exercise - Support Vector Machines

## Instructions:

### 1. Capture camera images
Use command `ffmpeg -i output.avi -f image2 output=%04d.png` to split video (output.avi) into image frames

### 2. Extract 50x50 cropped windows
Create output folder to store 50x50 pixels training images, e.g. `mkdir img50x50`
Edit lines 9 and 10 of `extract50x50.py` to specify input and output paths
Run `extract50x50.py` to extract 50x50 pixel images that has at least [min_red_pixels] red pixels (Note: change this threshold at line 24)

### 3. Manually classify training data
Manually sort images into three folders - 0-nosign, 1-stop and 2-warn.
Discard any ambiguous or irrelevant images to maintain clean training set.
 
### 4. Prepare data for processing
Run `prepdata.py` to generate pickle file data.p

### 5. Train using Support Vector Machines (SVM)
Run `signs-svm.py` to train and test svm. 
Note: trained model is saved as pickle file `model_svm.p`, while normalization parameters are stored in 'svm_params.p'

### 6. Sign Detection Pipeline
Run `pipeline.py` to process a single video avi file (`testmode='images'`) or a directory of captured camera image frames (`testmode='video'`)
Note: edit input paths in line 27 for video and line 25 for images
