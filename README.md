# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains starting files for the Behavioral Cloning Project.
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


My project includes the following files:
* *model.py* containing the script to create and train the model. *model.ipynb* containing a jupyter notebook version of the same script, with step-by-step execution results.
* *drive.py* for driving the car in autonomous mode
* A folder *models* containing trained neural network models:
** *model_track1.h5* containing a trained convolution neural network for driving on track 1 
** *model_track2.h5* containing a trained convolution neural network for driving on track 2

Due to large size of the networks (>100MB), they are stored off Github at the following Google Drive:
https://drive.google.com/open?id=1V4K-CFBg6iKs1h3wL1It0ZPhPnT-WYes

* Videos demonstration of the trained models on track 1 and 2.
** *run1.mp4* containing video for *model_track1.h5* driving on track 1 at speed 9.
** *run2.mp4* containing video for *model_track1.h5* driving on track 1 at speed 30.
** *track2.mp4* containing video for *model_track2.h5* driving on track 2 at speed 5.

* writeup_report.md summarizing the results

To test the pretrained model, download the trained models from [https://drive.google.com/open?id=1V4K-CFBg6iKs1h3wL1It0ZPhPnT-WYes]
to *models* directory.

Using the Udacity provided simulator and *drive.py* file, the car can be driven autonomously around the track by executing 
```sh
python drive.py ./models/model_track1.h5
```

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to create an architecture with multiple statges of convolution, pooling and non-linearity in order to extract features from images. Then, the extracted features are flatten and fed into a classifier, which is a fully-connected feed-forward neural network. 

My first step was to use a convolution neural network model similar to the model I used for the traffic sign classification task which worked quite well. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. Training and validation loss (MSE) is monitored during the training process to ensure that the optimizer works properly, and that the validation loss closely follows the training loss. The next step was to run the simulator to see how well the car was driving around track, observe any potentiall erratic behaviours.

This process was re-iterated many times to improve the model. I have found the following problem and fixes:

- Collecting more data persistently imrpoves the model quality. In the end, I colleced ~40k of images by manually driving around the track both clockwise and counter-clockwise.

- There are a large number of images with zero steering angle. In order to balance the training data, we randomly keep only 50% of the images with zero-steering angle. I have found that this selection speeds up traning while also improving the model driving behaviour.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines xx-xx) consisted of a convolution neural network with the following layers and layer sizes.

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

To capture good driving behavior, I first recorded 4 laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][center]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover.

![alt text][left]
![alt text][right]


Since the two tracks are markedly different, I decided not to collect data on track 2 for the track 1 driving part.

To augment the data set, I also flipped images and steering angles.
![alt text][flip]


The images were cropped, eliminating the top 50 pixels and bottom 20 pixels which contains information that is not essential to predict the steering angle (e.g., the sky and bonnet).

After the collection process, I had ~28k data points. I then preprocessed this data by rescaling the data to the range [-1, 1].


I finally randomly shuffled the data set and put 5% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used an SGD optimizer with momentum. The learning rate was set to 0.001. Choosing larger learning rates led to unstablized training with *nan* loss. To encourage the model to settle into a local minimum, I also lowered the training rate to 0.0001 and further refine the model for 100 epochs. 

### Track 2 driving
Due to markedly different road conditions, the model trained on track 1 failed to drive on track 2. In order to drive on track 2, we finetune this model by collecting ~25k images of driving practice on track 2. The model is finetuned for 50 epochs. Upon completing this finetuning, the model can drive on track 2 for an extended distance, as seen in the *track2.mp4* video.


