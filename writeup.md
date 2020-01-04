# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* [x] Load the data set (see below for links to the project data set)
* [x] Explore, summarize and visualize the data set
* [x] Design, train and test a model architecture
* [x] Use the model to make predictions on new images
* [x] Analyze the softmax probabilities of the new images
* [x] Summarize the results with a written report


[//]: # (Image References)

[image1]: ./img/hist_t.png "Histogram Train"
[image2]: ./img/hist_v.png "Histogram Validation"
[image3]: ./img/hist_test.png "Histogram Test"
[image4]: ./signs/dverb.jpg "Traffic Sign 1"
[image5]: ./signs/30.jpg "Traffic Sign 2"
[image6]: ./signs/vfahrt.png "Traffic Sign 3"
[image7]: ./signs/1bahn.jpg "Traffic Sign 4"
[image8]: ./signs/100.jpg "Traffic Sign 5"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Data Set Summary & Exploration

#### 1. Basic summary of the data set

I used the pandas and numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

The three histograms showing the distributions of all labels in the train, validation and test set.

![alt text][image1]
![alt text][image2]
![alt text][image3]

### Design and Test a Model Architecture

#### 1. Preprocessed the image data

+ I did not use an conversion to grayscale, because the color of the sign is a useful information for classification
+ I normelized the data via X_train/ 255 * 0.8 + 0.1
+ I then used sklearn.utils to shuffle the training data


#### 2. Final model architecture 

My final model i used a LeNet consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 1     	| 1x1 stride, VALID padding, output = 28x28x6	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, VALID padding, output = 14x14x6	|
| Convolution 2 	    | 1x1 stride, VALID padding, output = 10x10x16	|
| RELU          		|           									|
| Max pooling			| 2x2 stride, VALID padding, output = 5x5x16	|
| Flatten				| output = 400                                  |
|  Fully connected 		| input = 400, output = 120	                    | 
|  RELU 		        |                                               | 
|  Fully connected 		| input = 120, output = 84                      | 
|  RELU 		        |                                               | 
|  Fully connected 		| input = 84, output = 43                       | 
 

#### 3. Description how I trained the model.

To train the model, I used:
+ EPOCHS = 20
+ BATCH_SIZE = 128
+ learning rate = 0.001
+ cross_entropy as optimizer (tf.nn.softmax_cross_entropy_with_logits)
+ AdamOptimizer

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.901 
* test set accuracy of 0.869

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. five German traffic signs found on the 

Here are five German traffic signs that I found on the web:

![alt text][image4] 
![alt text][image5]
![alt text][image6] 
![alt text][image7] 
![alt text][image8]

The first image might be difficult to classify because its not in the list of classes.
The rest should be fine!

#### 2. model's predictions on these new traffic signs 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| transit prohibited 	| Roundabout mandatory							| 
| Speed limit (30km/h)	| Speed limit (30km/h)							|
| Priority road			| Priority road									|
| No entry	      		| No entry				 				        |
| Speed limit (100km/h)	| Right-of-way at the next intersection  		|

The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. 
Its no wonder a not trained sign results in a false class but I don't know why the Speed limit (100km/h)
was not correctly classified...

#### 3. How certain the model prediction on each of the five new images

+ transit prohibited 
  + Roundabout mandatory: 99.03%
  + Traffic signals: 0.82%
  + General caution: 0.13%
  + End of all speed and passing limits: 0.01%
  + Keep right: 0.01%  
  
+ Speed limit (30km/h)
  + Speed limit (30km/h): 100.00%
  + Speed limit (50km/h): 0.00%
  + Speed limit (60km/h): 0.00%
  + Speed limit (20km/h): 0.00%
  + Yield: 0.00%

+ Priority road
  + Priority road: 100.00%
  + No entry: 0.00%
  + No passing: 0.00%
  + Stop: 0.00%
  + Yield: 0.00%

+ No entry
  + No entry: 100.00%
  + Speed limit (20km/h): 0.00%
  + Slippery road: 0.00%
  + Stop: 0.00%
  + Dangerous curve to the right: 0.00%

+ Speed limit (30km/h)
  + Right-of-way at the next intersection: 99.98%
  + Pedestrians: 0.01%
  + Roundabout mandatory: 0.00%
  + Speed limit (30km/h): 0.00%
  + Speed limit (50km/h): 0.00%