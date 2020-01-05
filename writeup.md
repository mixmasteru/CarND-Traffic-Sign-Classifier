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
+ I then used sklearn.utils to shuffle the training data
+ I used train_test_split from sklearn.model_selection to split the data
+ I normalized the data via X_train/ 255 * 0.8 + 0.1

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

#### 4. validation set accuracy to be at least 0.93.

I stared with the Lenet DNN from the lessons and got good results from the beginning. With the given data split I could 
not archive the necessary accuracy for the validation set, so I did split the training data and used them.

My final model results were:
* validation set accuracy of 0.978 
* test set accuracy of 0.894

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

#### 2. Model's predictions on these new traffic signs 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| transit prohibited 	| Roundabout mandatory							| 
| Speed limit (30km/h)	| Speed limit (30km/h)							|
| Priority road			| Priority road									|
| No entry	      		| No entry				 				        |
| Speed limit (100km/h)	| Speed limit (30km/h) 	                    	|

The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. 
Its no wonder a not trained sign results, the Speed limit (100km/h) was not correctly but as Speed limit (30km/h) 
which is quite similar 

#### 3. The model prediction on each of the five new images

+ transit prohibited 
  + Ahead only: 52.32%
  + Speed limit (50km/h): 46.24%
  + Keep right: 1.37%
  + Yield: 0.06%
  + Speed limit (30km/h): 0.00%
  
+ Speed limit (30km/h)
  + Speed limit (30km/h): 99.87%
  + Speed limit (20km/h): 0.13%
  + Speed limit (50km/h): 0.00%
  + Speed limit (70km/h): 0.00%
  + Keep right: 0.00%

+ Priority road
  + No entry: 100.00%
  + Stop: 0.00%
  + Traffic signals: 0.00%
  + Slippery road: 0.00%
  + Bumpy road: 0.00%

+ No entry
  + Priority road: 100.00%
  + Stop: 0.00%
  + End of no passing by vehicles over 3.5 metric tons: 0.00%
  + No passing: 0.00%
  + No passing for vehicles over 3.5 metric tons: 0.00%

+ Speed limit (30km/h)
  + Speed limit (30km/h): 89.67%
  + Roundabout mandatory: 10.33%
  + Speed limit (20km/h): 0.00%
  + Speed limit (50km/h): 0.00%
  + Speed limit (70km/h): 0.00%