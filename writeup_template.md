#**Traffic Sign Recognition** 

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./plots/data_viz.png "Visualization"
[image2]: ./plots/preproc_data_viz.png "Grayscaling"
[image3]: ./plots/yield_13.jpg "Yield Sign"
[image4]: ./plots/speed_limit_70_4.png "70 Sign"
[image5]: ./plots/slippery_road_23.jpg "Slippery Road Sign"
[image6]: ./plots/road_work_25.jpg "Road Work Sign"
[image7]: ./plots/road_narrows_right_24.jpeg "Road Narrows Right Sign"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/haallen/CarND-Traffic-Sign-Classifier-P2/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is located in Step 1 of the notebook.  

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

For each class, I also determined the fraction of the training set represented by that class. 
The data is as follows:

0 0.00517256242995
1 0.0568981867295
2 0.0577602804678
3 0.0362079370097
4 0.0508635305612
5 0.0474151556079
6 0.0103451248599
7 0.037070030748
8 0.0362079370097
9 0.0379321244863
10 0.0517256242995
11 0.0336216557947
12 0.0543119055145
13 0.0551739992529
14 0.0198281559815
15 0.0155176872899
16 0.0103451248599
17 0.0284490933648
18 0.0310353745797
19 0.00517256242995
20 0.00862093738326
21 0.00775884364493
22 0.00948303112158
23 0.0129314060749
24 0.00689674990661
25 0.0387942182247
26 0.0155176872899
27 0.00603465616828
28 0.0137934998132
29 0.00689674990661
30 0.0112072185982
31 0.0198281559815
32 0.00603465616828
33 0.0172131383086
34 0.0103451248599
35 0.0310353745797
36 0.00948303112158
37 0.00517256242995
38 0.0534498117762
39 0.00775884364493
40 0.00862093738326
41 0.00603465616828
42 0.00603465616828

TODO: figure out how to display the above list properly in markdown.

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is also contained in Step 1 of the notebook.  

Here is an exploratory visualization of the data set. It is 5 images selected at random from the training set and plotted in color. The title of each image corresponds to its label.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in Step 2 of the notebook.

For each of the training images:
 * convert to greyscale
 * normalize the greyscale image by dividing each pixel value by 255 and subtracting 0.5
    * results in pixel values between -0.5 and +0.5
 * append the normalized greyscale image to a list of preprocessed training images

I tried different preprocessing techniques but found that the selected approach performed the best for my chosen neural network architecture. I would like to revisit this at some point; I feel processing the RGB or even the YUV image could lead to potentially better results.

The above steps were repeated for validation and test data.

For each the 5 images that were previously plotted, I plotted the preprocessed version of the image as a sanity check.
![alt text][image2]

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

Training, Validation, and Test data were provided to me. I loaded the provided data in Step 0 of the notebook. A Summarization of the data is provided earlier in this writeup.

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the Model Architecture cell of Step 2 of the notebook. 

The overall structure of my selected model is roughly that of the provided Lenet architecture.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 greyscale image   							| 
| Convolution 5x5x1x6     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|							| 
| Convolution 5x5x1x16     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				| 									|
| Fully connected		| output = 120        									|
| RELU					|			
| Dropout					|	keep prob = 50% (configurable)
| Fully connected		| output = 84       									|
| RELU					|			
| Dropout					|	keep prob = 50% (configurable)
| Fully connected		| output = 10       									|

####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the 'Train, Validate and Test the Model' cells of Step 2 of the notebook. 

Training consisted of 40 epochs. For each epoch, I first shuffled all the training data divided the training data into batches of 128. For each batch, the error between the predicted classification and actual classification was calculated using the softmax_cross_entropy_with_logits function. The average cross entropy loss was then minimized using the AdamOptimizer. The values for the learning rate and dropout probability were found by trying a few values and selecting the values that resulted in the best performance.

After all batches in an epoch were trained, the model was then evaluated against both the training data and the validation data and then these values, along with the corresponding loss were printed to the screen. This process was then repeated for the specified number of epochs. Once I decided on my final architecture, I evaluated the model against the test data.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is also located in the 'Train, Validate and Test the Model' cells of Step 2 of the notebook. 

My final model results were:
* training set accuracy of 0.997
* validation set accuracy of 0.954
* test set accuracy of 0.934

I arrived at this through an iterative approach.

I trained the provided Lenet-5 architecture without any adjustments other than increasing the number epochs and found that it performed decently well against the training data, but not the validation data. This led me to believe that it was overfitting to the training data. I added in dropout layers and tried different values for the keep_prob. Overall, dropout increased the validation accuracty  but there didn't seem to be a signficant advantage of having many dropout layers or varying the keep_prob much. Based on the class videos, I felt like I should stay with the provided Lenet-5 architecture and try to optimize for the data. 

I then tried to optimize the learning rate. It was initially too large, the loss would decrease but then more or less plateau. I tried different values of the learning rate and my selected rate seemed to be a good balance between the results and the amount of training time.

I also modified my architecture and preprocessing step to try to train on the original color images as well as the images converted to YUV. Perhaps I had an error in my approach, but I found that the model and preprocessing for the greyscale images outperformed the other approaches and was much more simple.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image3] ![alt text][image4] ![alt text][image5] 
![alt text][image6] ![alt text][image7]

I selected the images based on the frequency that they appear in the training data as well as features about the signs

The first image, Yield, should be easy to classify because it appears frequently in the training set and the doesn't have a lot of detailed features. The second image, 70 km/h speed limit, also appears frequently but I was curious to see if my model could correctly classify the numbers on the sign. The third and fourth signs, slippery and road work, appears much less frequently than the first 2 and also have interesting, detailed features. The last sign, road narrows on right, appears the least often of the bunch and also shares overlapping major features with the 3rd and 4th signs (red triangles with white center and black images in them). These major features are also common to a lot of the other signs in the set. I wouldn't be surprised if my classifier had troubles with them.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in Step 3 of the notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Yield       		| Yield  									| 
| 70 km/h    			| 70 km/h										|
| Slippery Road					| General Caution												|
| Road Work	      		| 	Road Work			 				|
| Road Narrows Right				| Road Narrows Right	      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. The accuracy performance against the test set (93%) is much higher, but I would need to pull a lot more images from the web to do a fair comparison. 4 out of 5 images correctly classified doesn't seem that bad.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in Step 3 of the notebook.

Here's the output. 
Sign | top 5 probabilities | class for each of the top 5 probabilites
Yield Sign |1.00E+00	9.22E-22	2.10E-24	2.07E-28	6.26E-29 |		13	35	38	36	10
70 km/h Sign |1.00E+00	1.34E-06	2.41E-11	7.42E-15	3.50E-15	|	4	1	0	14	40
Slippery Sign | 1.00E+00	1.41E-09	2.84E-12	3.01E-15	1.44E-15|		18	25	11	20	38
Road Work Sign |9.66E-01	3.41E-02	6.26E-05	1.27E-05	4.85E-06|		25	22	30	20	38
Road Narrows Right Sign | 6.68E-01	1.59E-01	7.78E-02	2.37E-02	2.33E-02 |		24	27	11	21	30

The model is very certain (near 100%) in its classification for the first 3 images, reasonably certain (97%) in the fourth image, and not very certain (67%) in the last image. The difference between the probability for the selected image and the probability for the next choice also corresponds to this pattern for the 5 images (very large difference between first and second probability for the first three images, small difference for the last image, etc.)

The two images that are of interest to me are the 3rd and 5th images. 

For the third image, the slippery sign, the model was very certain in its classification but the classification was incorrect. It is interesting that the correct class doesn't show up at all in the top 5 probabilities. The selected sign, the general caution sign, does not really appear to have many of the same features as the slippery sign, other than the main shape/color features. If I could get the visualizations (step 4 of the notebook) working, I would try to visualize some of the layers for this to see what is happening.

For the fifth image, the road narrows on right sign, the model was not that confident in its classification but ultimately got it right. The second choice, the pedestrian sign, is surprising because in color, the images do not look similiar, but in greyscale they could potentially share features. Both the road narrows on right sign and the pedestrian sign do not appear frequently in the training set; by augmenting the training set with additional data and retraining, perhaps this issue could be mitigated.
