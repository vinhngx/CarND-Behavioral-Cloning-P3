# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)
[center]: ./examples/center.jpg "Center driving"
[left]: ./examples/left.jpg "Recovery Image"
[right]: ./examples/right.jpg "Recovery Image"
[flip]: ./examples/center_flip.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* *model.py* containing the script to create and train the model. *model.ipynb* containing a jupyter notebook version of the same script, with step-by-step execution results.
* *drive.py* for driving the car in autonomous mode
* A folder *models* containing trained neural network models:
  * *model_track1.h5* containing a trained convolution neural network for driving on track 1 
  * *model_track2.h5* containing a trained convolution neural network for driving on track 2

Due to large size of the networks exceeding Github limit (>100MB each), they are stored off Github at the following Google Drive:
https://drive.google.com/open?id=1V4K-CFBg6iKs1h3wL1It0ZPhPnT-WYes

* Videos demonstration of the trained models on track 1 and 2.
  * *track1_speed9.mp4* containing video for *model_track1.h5* driving on track 1 at speed 9.
  * *track1_speed30.mp4* containing video for *model_track1.h5* driving on track 1 at speed 30.
  * *track2.mp4* containing video for *model_track2.h5* driving on track 2 at speed 5.

* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Download the trained models from https://drive.google.com/open?id=1V4K-CFBg6iKs1h3wL1It0ZPhPnT-WYes
to *models* directory.

Using the Udacity provided simulator and *drive.py* file, the car can be driven autonomously around the track by executing 
```sh
python drive.py ./models/model_track1.h5
```

The *drive.py* script can be modified supplying different driving speed. We observe that at higher driving speeds it is generally more challenging to keep the car on track.

#### 3. Submission code is usable and readable

The *model.py* file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains brief comments to explain how the code works.

See *model.ipynb* for a step by step run and outcome of the same script. 

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py). The model includes RELU layers to introduce nonlinearity.

The data is normalized in the data generator to range within [-1,1]. As per suggestion in the lecture, the image is further cropped to exclude the top 50 and bottom 20 rows, which contains information not esstentiall to driving, such as the sky and the car hood. 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in between the fully connected layers in order to reduce overfitting (model.py lines 177). 

Furthermore, we employ L2 regularization for every layer with weights (i.e., convolutional or fully connected) in order to counteract overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 105-106 where we generate separate data generators for train and validation sets). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used a stochastic gradient descent optimizer with momentum. The learning rate was set to 0.001 for 100 epoch. We observe that if the learning rate is set to higher values like 0.01, sometimes the training process diverges with *nan* loss values.

Next, we finetune the model with learnign rate 0.0001 for another 100 epochs to encourage the model to settle near a local minimum. 

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to create an architecture with multiple statges of convolution, pooling and non-linearity in order to extract features from images. Then, the extracted features are flatten and fed into a classifier, which is a fully-connected feed-forward neural network. 

My first step was to use a convolution neural network model similar to the model I used for the traffic sign classification project which worked quite well. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. Training and validation loss (MSE) is monitored during the training process to ensure that the optimizer works properly, and that the validation loss closely follows the training loss. The next step was to run the simulator to see how well the car was driving around track, observe any potentiall erratic behaviours.

This process was re-iterated many times to improve the model. I have found the following problem and fixes:

- Collecting more data persistently imrpoves the model quality. In the end, I colleced ~40k of images by manually driving around the track both clockwise and counter-clockwise.

- There are a large number of images with zero steering angle. In order to balance the training data, we randomly keep only 50% of the images with zero-steering angle. I have found that this selection speeds up traning while also improving the model driving behaviour.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 127-180) consisted of a convolution neural network with the following layers and layer sizes.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   						| 
| Cropping         		| 90x320x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 90x320x16 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 90x320x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 45x160x32 				|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 45x160x32 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 45x160x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 22x80x64 				|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 22x80x64 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 22x80x128 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 11x40x128 				|
| Flatten | Output 56320 |
| Fully connected | 256 |
| Dropout | keep_prob = 0.5 |
| Fully connected | 128 |
| Dropout | keep_prob = 0.5 |
| Fully connected		| output 1	|

The total number of trainable parameters is 14,594,817.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded 4 laps on track 1 using center lane driving. Here is an example image of center lane driving:

![alt text][center]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover.

![alt text][left]
![alt text][right]


Since the two tracks are markedly different, I decided not to collect data on track 2 for the track 1 driving part.

To augment the data set, I also flipped images and steering angles.
![center][center]
![flip][flip]

The images were cropped, eliminating the top 50 pixels and bottom 20 pixels which contains information that is not essential to predict the steering angle (e.g., the sky and bonnet).

To capture further training data, I also drove around the track in the reverse direction for another 4 laps. 

After the collection process, I had ~40k data points. I then preprocessed this data by rescaling the data to the range [-1, 1].


I finally randomly shuffled the data set and put 5% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used an SGD optimizer with momentum. The learning rate was set to 0.001. Choosing larger learning rates led to unstablized training with *nan* loss. To encourage the model to settle into a local minimum, I also lowered the training rate to 0.0001 and further refine the model for 100 epochs. 

### Track 2 driving
Due to markedly different road conditions, the model trained on track 1 failed to drive on track 2. In order to drive on track 2, we finetune this model by collecting ~25k images of driving practice on track 2. The model is finetuned for 50 epochs with SGD at learning rate 0.001. Upon completing this finetuning, the model can drive on track 2 for an extended distance, as seen in the *track2.mp4* video.






