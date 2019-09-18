# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


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

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

### Data Set Summary & Exploration

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

Here is an exploratory visualization of the data set. It is a bar chart showing the data distribution.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. 
As a first step, I decided to convert the images to grayscale because using a single color channel input will expedite the training 

process. Besides that, using three color channel does not guarantee better performance and doing so will consume significantly more 

computational power.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image3]

As a last step, I normalized the image data so that data will have the same feature distribution range. By doing so, it allows the network 

to learn faster.

I decided to generate additional data because the number of data for each class varies greatly. Little amount of data in one class will 

cause the model fail to classify classes with lesser data.

To add more data to the the data set, I apply random rotation, brightness adjustment, random affine transformation, and random translation.

Here is an example of an original image and an augmented image:

![alt text][image4]

Here is another bar chart showing the distrubution of augmented data:

![alt text][image2]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

![alt text][image9]


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer to train my network. I also used following parameters:

BATCH SIZE: 128
Number of EPOCHS: 60
Learning rate: 0.001

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 97.7%
* test set accuracy of 96.5%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
The first architecture that was tried is the original LeNet. It is choosen because it works great on MNIST data set which is also 32x32 pixels images.

* What were some problems with the initial architecture?
I was only able to achieve 89.16% accuracy on validation data using the original LeNet.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
I replaced few layers of the original LeNet with an inception module and achieve around 92% accuracy. After that, I took away few layers of 1x1 convolution as in my opinion, applying a 1x1 conv on a layers with dept of 6 seems to be redundant.

* Which parameters were tuned? How were they adjusted and why?
The learning rate and number of Epochs are adjusted by a trial and error method. The model is trained several time and the learning rate and number of Epochs are changed by a small amount in each run.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
The dropout layer is one of the most important design choices. After adding the dropout layer with 50% percent of drop out rate, the accuracy has increased by 4%. The dropout layer might have prevent the model from overfitting as it forces the model to depend on numerous neurons.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set .

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 70 km/h       		| 20 km/h   									| 
| Keep Left     		| Keep Left 									|
| Roundabout			| Roundabout									|
| No Entry      		| No Entry 					 					|
| Children Crossing		| Children Crossing      						|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 96.5%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|  1.         			| 20 km/h   									| 
|  0.     				| 30 km/h 										|
|  0.					| 50 km/h										|
|  0.	      			| 60 km/h  						 				|
|  0.				    | 70 km/h        								|


For the second image:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|  1.         			| Keep Left   									| 
|  0.     				| 20 km/h 										|
|  0.					| 30 km/h										|
|  0.	      			| 50 km/h  						 				|
|  0.				    | 60 km/h        								|


For the third image:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|  1.         			| Roundabout   									| 
|  2.09142546e-16 		| 100 km/h 										|
|  1.79696749e-17		| 120 km/h										|
|  1.26221117e-21		| No Vehicles  					 				|
|  1.25843018e-32		| 70 km/h        								|


For the fourth image:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|  1.         			| No Entry   									| 
|  0.     				| 20 km/h 										|
|  0.					| 30 km/h										|
|  0.	      			| 50 km/h  						 				|
|  0.				    | 60 km/h        								|


For the fifth image:
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|  1.         			| Children Crossing  							| 
|  0.     				| 20 km/h 										|
|  0.					| 30 km/h										|
|  0.	      			| 50 km/h  						 				|
|  0.				    | 60 km/h        								|
