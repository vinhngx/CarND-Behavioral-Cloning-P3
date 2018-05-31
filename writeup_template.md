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

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

See model.ipynb for a step by step run and outcome of the script. 

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py). The model includes RELU layers to introduce nonlinearity.

The data is normalized in the data generator to range within [-1,1]. The image is further cropped to exclude the top 50 and bottom 20 rows. 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in between the fully connected layers in order to reduce overfitting (model.py lines 21). 

Furthermore, we employ L2 regularization for every layer with weights (i.e., convolutional or fully connected) in order to counteract overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used a stochastic gradient descent optimizer with momentum. The learning rate was set to 0.01 for 100 epoch. This learning rate is gradually decayed to ensure convergence to a local optimum.  

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to create an architecture with multiple statges of convolution, pooling and non-linearity in order to extract features from images. Then, the extracted features are flatten and fed into a classifier, which is a fully-connected feed-forward neural network. 

My first step was to use a convolution neural network model similar to the model I used for the traffic sign classification task which worked quite well. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. Training and validation loss (MSE) is monitored during the training process to ensure that the optimizer works properly, and that the validation loss closely follows the training loss. The next step was to run the simulator to see how well the car was driving around track, observe any potentiall erratic behaviours.

This process was re-iterated many times to improve the model. I have found the following problem and fixes:

- Collecting more data persistently imrpoves the model quality. In the end, I colleced ~40k of images by manually driving around the track both clockwise and counter-clockwise.

- There are a large number of images with zero steering angle. In order to balance the training data, we randomly keep only 20% of the images with zero-steering angle. I have found that this selection speeds up traning while also improving the model driving behaviour.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   							| 
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

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover.

![alt text][image3]
![alt text][image4]
![alt text][image5]

Since the two tracks are markedly different, I decided not to collect data on track 2.

To augment the data sat, I also flipped images and steering angles.
The images were cropped, eliminating the top 50 pixels and bottom 20 pixels which contains information that is not essential to predict the steering angle (e.g., the sky).

After the collection process, I had ~28k data points. I then preprocessed this data by rescaling the data to the range [-1, 1].


I finally randomly shuffled the data set and put 5% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 100 as evidenced by the flattening loss. I used an SGD optimizer with momentum. The learning rate was set to 0.01. Choosing larger learning rates led to unstablized training.
