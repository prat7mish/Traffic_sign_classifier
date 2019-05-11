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


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! 

Refer Traffic_Sign_Classifier.ipynb for the main code and logic. Also the "test_images" folder for sample images used for testing purposes.
"report.html"- contains the HTML export of python notebook.


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually. (Refer cell 2)

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset. (Refer cells 3-4)

We have used used scipy statistical functions and matplotlib.pyplot plotting framework.

It can be observed that the training, validation and testing datasets have a similar number of examples per label distribution.

This can help understand where potential pitfalls could occur if the dataset isn't uniform in terms of a baseline occurrence

Further I have displayed one of the sample output of the image observed along with the label that corresponds to the traffic sign number in the CSV(signnames.csv) used for reference.


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (Refer cell 5)

As a first step, I have shuffled the images so that everytime the training that happens is not biased in nature.

Next, I decided to scale the images so that the data values are between -1 and 1.
Standardizing either input or target variables tends to make the training process better behaved
Here by normalizing the images by scaling them also by the minimum and range of the vector is done to make all the elements lie between 0 and 1. In our case, we have chosen to scale between -1 and 1.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.(Refer cell 7)

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution       	| Input = 32x32x3. Output = 28x28x6         	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 14x14x6   				|
| Convolution   	    | Input: Input = 14x14x6. Output = 10x10x16.    |
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 5x5x16    				|
| Flatten				|Input = 5x5x16. Output = 400.					|
| Fully connected		|Input = 400. Output = 300. 					|
| RELU					|												|
| Fully connected		|Input = 300. Output = 200. 					|
| RELU					|												|
| Fully connected		|Input = 200. Output = 120. 					|
| RELU					|												|
| Fully connected		|Input = 120. Output = 84.  					|
| RELU					|												|
| Dropout				|0.8    										|
| Fully connected		|Input = 84. Output = 60.   					|
| RELU					|												|
| Dropout				|0.8    										|
| Fully connected		|Input = 64. Output = 32.   					|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.(Refer cell 8)

During the training step, the Adam optimizer is used to try to minimize the loss. Loss is calculated by reducing the mean of the cross entropy function, which uses the Softmax function to predict the output between the different 43 categories.

The following hyperparameters have been tuned to improve the model accuracy:

1. learning rate
2. epochs
3. batch size

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem. (Refer cell 9-11)

My final model returned a 0.937 accuracy on the validation set, and a 0.939 accuracy on the testing set.

These were the steps I followed to get the results above:

I first trained the original LeNet architecture with the specifications as shown in the tutorial videos.

Then I started playing different combinations of epochs, batch_size and learning rate.
I tried to change the epochs in the values of 20,40,60,80--- 40 being the best performant.I further improved it by increasing epochs from 40 to 45 and found 44 to be my final epoch value.
I decreased the learning rate from 0.1 to 0.001, trying 0.01, 0.005 and 0.002. 0.001 was the most performant in term of accuracy.
Decreased the batch size from 128 to 90 to 60, 60 which was the selected batch size because it was the most performant in terms of accuracy.
With the changes above I obtained an increase of the accuracy of around 0.3.

Then I started adding fully connected layers and dropout until getting the final solution.

Please notice that the second max_pool layer from the original LeNet architecture has been replaced by an avg_pool layer. It increased the accuracy (around 0.1) so I decided to keep it.

When I decided to increase the number of fully connected layers, I was concerned to overfit the network. That's the reason why I added two dropout layers in the final steps.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify. (Refer cell 12-13)

The five German traffic signs that I found on the web can be found to be contained in the "test_images" folder at the location: "CarND-Traffic-Sign-Classifier-Project/test_images/"

Please notice that each of the images had a different size. They were rescaled programmatically when loaded.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric). (Refer cell 14 onwards)

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (70km/h)  | Speed limit (50km/h) 							| 
| Turn right ahead   	| Turn right ahead 								|
| Stop					| Stop											|
| Speed limit (60km/h)	| Speed limit (60km/h)					 		|
| Speed limit (50km/h)	| Speed limit (50km/h)      					|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%.
