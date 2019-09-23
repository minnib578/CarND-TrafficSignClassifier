# **Traffic Sign Recognition** 
---

[//]: # (Image References)

[image1]: ./BeforeVisualization.png "Before Augmentation Visualization"
[image2]: ./AfterVisualization.png "After Augmentation Visualization"
[image2]: ./grayscale.jpg "Grayscaling"
[image3]: ./augmentation.jpg "Augmented Data"
[image4]: ./Image001.jpg "Traffic Sign 1"
[image5]: ./Image002.jpg "Traffic Sign 2"
[image6]: ./Image003.jpg "Traffic Sign 3"
[image7]: ./Image004.jpg "Traffic Sign 4"
[image8]: ./Image005.jpg "Traffic Sign 5"
[image9]: ./NN_Architecture.jpg "NN Architecture"

## Introduction

_**Note:** This project makes use of  [Udacity Traffic Sign Classifier Project Repository]( https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project) and [German Traffic Sign Dataset]( http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). The dataset can be found in the link above._

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Perform data augmentation
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

In this project,
---
### Data Set Summary & Exploration

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

Here is an exploratory visualization of the data set. It is a bar chart showing the data distribution.

![alt text][image1]

### Data Augmentation

First, images augmentation is performed to generate more data for the training phase. Four data augmentation functions are introduced in the code which are random translation, brightness, rotation, and affine transformation. Little amount of data in certain classes will cause the model failing to classify classes with less data. Since the data distribution is not even, data categories with less than 800 images are augmentated to generate more data and achieve more even data distribution.

Here is an example of an original image and an augmented image:

![alt text][image4]

The augmentated data distribution chart is shown below:

![alt text][image2]

### Data Preprocessing

Before the data are fed into the network, the images are first converted to grayscale because using a single color channel input will expedite the training process. Besides that, using three color channel does not guarantee better performance and doing so will consume significantly more computational power.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image3]

Next, the grayscale images are then normalized so that the data have the same feature distribution range. By doing so, it allows the network to learn faster.

### Network Architecture

My final model consisted of the following layers:

![alt text][image9]

The following model is based on classic LeNet and modified to achieve better result.
The initial architecture tried is the original LeNet. It is choosen because it works great on MNIST data set which is also 32x32 pixels images. However, the highest accuracy achieved with this architecture is only 89.16% on validation data. Therefore, the model is then modified by adding branch layers in the middle of the networks and then the outputs are concatenated into fully connected layers.
50% dropout layers are also added to the fully connected layers to prevent the model from overfitting. Adding the dropout layers do help increasing the accuracy by 4%.
The model is trained using Adam Optimizer and following parameters:

BATCH_SIZE: 128
EPOCHS: 60
Learning_rate: 0.001

### Results

My final model results were:
* validation set accuracy of 97.7%
* test set accuracy of 96.5%

#### Test a Model on New Images

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 70 km/h       		| 20 km/h   									| 
| Keep Left     		| Keep Left 									|
| Roundabout			| Roundabout									|
| No Entry      		| No Entry 					 					|
| Children Crossing		| Children Crossing      						|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 96.5%.