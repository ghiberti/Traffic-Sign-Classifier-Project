# **Traffic Sign Recognition** 

## Writeup

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

[image1]: ./examples/barchart.png "Bar chart"
[image2]: ./examples/color.png "Original image"
[image3]: ./examples/grayscale.png "Grayscale image"
[image4]: ./data/examples/keep_right.jpg "Keep Right"
[image5]: ./data/examples/limit_70.jpg "Limit 70"
[image6]: ./data/examples/priority.jpg "Priority Road"
[image7]: ./data/examples/roundabout.jpg "Roundabout"
[image8]: ./data/examples/stop.jpg "Stop"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/ghiberti/Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* Number of training examples = 34799
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43


#### 2. Include an exploratory visualization of the dataset.

The bar chart displays the distribution of data for each traffic sign class in the training set. Lopsided distribution of training examples could point to the fact that the trained model may not be able to generalize equally well for all trained classes in the real world. 

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I used grayscale images for training because the color aspect of the data is not relevant to the recognition task, therefore reducing the three channels to one will make the model simpler thus the training and inference operations can be done faster. 

Here are an examples of a traffic sign image before and after grayscaling.

![alt text][image2] ![alt text][image3]

Secondly, I applied normalization to the data to make sure all data are fit in the same range. This allows the network to train faster and allow model weights to deal with data on the same scale. 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 7x7     	| 2x2 stride, valid padding, outputs 13x13x16 	|
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 5x5x32 	|
| RELU					|												|
| Convolution 3x3     	| 2x2 stride, valid padding, outputs 2x2x64 	|
| RELU					|												|
| Fully Connected    	| 256 nodes                                 	|
| Dropout				|												|
| Fully Connected    	| 80 nodes                                     	|
| Dropout				|												|
| Fully Connected    	| 43 nodes in final layer for each class      	|
| Softmax   			|												|

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I used the following hyperparameters together with Adam Optimizer for training the model:

epochs = 20
batch size = 512
dropout keep probability = 0.5
learning rate = 0.001

I decided the number of epochs and Adam Optimizer after multiple trial runs. Adam optimizer performed best for fastest convergence, and validation accuracy appeared to stabilize after 20 epochs.

Regarding batch size, dropout keep probability and learning rate, I used figures that have been highly recommended by the deep learning community as a rule of thumb for a good baseline performance. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I took the baseline convolutional network given in the LeNet Lab class and tried to improve on it by increasing kernel size, switching maxpooling with (2,2) convolutional strides, using valid padding instead of same and using dropout for regularization.

I tried moving from bigger to smaller kernel size through convolution layers with the reasoning that important features would manifest at larger scales with larger images. Furthermore I used even numbers for kernel sizes to have central axes in convolution frame.

Replacing maxpooling with (2,2) convolutional strides was a recommend by Springenberg et al. in their 2014 paper "Striving for Simplicity: The All Convolutional Net".

Finally I used valid padding not to lose any possibly valuable pixels around image edges and used 0.5 probability dropout for optimal regularization.

After multiple runs experimenting with different Optimizers and numbers of epochs, I was finally satisfied with the result.

My final model results were:

* Training Accuracy = 0.986
* Validation Accuracy = 0.936
* Training Accuracy = 0.913 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

First three images should be relatively easier to classify due to greater number training examples existing in the training set for these signs. On the other hand roundabout and stop sign may be harder due to the same reason.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Keep right     		| Keep right  									| 
| Limit 70 km/h			| Limit 70 km/h									|
| Priority Road			| Priority Road									|
| Roundabout      		| Roundabout					 				|
| Stop          		| Stop                							|


The model was able to correctly guess all 5 examples. Results compare favoribly with the validation and test accuracy, possibly meaning that the trained model is able to generalize well to new input. This is especially more surprising for roundabout and stop signs which had much less training data compared to three other signs.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

First Image: Limit 70km/h

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .95         			| Limit 70km/h 									| 
| .03    				| Limit 20km/h 									|
| .007					| Limit 120km/h									|
| .007	      			| Limit 30km/h					 				|
| .001				    | Roundabout mandatory 							|


Second Image: Keep Right

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .100        			| Keep Right 									| 
| .000   				| No Vehicles 									|
| .000					| Dangerous curve to right						|
| .000	      			| Priority Road 				 				|
| .000				    | Yield               							|

Third Image: Priority Road

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99        			| Priority Road									| 
| .000   				| Roundabout mandatory							|
| .000					| Right-of-way at the next intersection			|
| .000	      			| General caution 				 				|
| .000				    | End of all speed and passing limits			|

Fourth Image: Roundabout mandatory

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .97         			| Roundabout mandatory							| 
| .02    				| Priority road									|
| .001  				| Limit 30km/h	    							|
| .000	      			| Go straight or left			 				|
| .000				    | Limit 100km/h        							|

Fifth Image: Stop

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .40        			| Stop       									| 
| .34   				| No entry   									|
| .14					| Keep right    								|
| .05	      			| Turn left ahead				 				|
| .03   			    | Priority road        							|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


