## Build a Traffic Sign Recognition Program

### Overview

In this project, you will use what you've learned about deep neural networks and convolutional neural networks to classify traffic signs. You will train a model so it can decode traffic signs from natural images by using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, you will then test your model program on new images of traffic signs you find on the web, or, if you're feeling adventurous pictures of traffic signs you find locally!

### Dependencies

This project requires **Python 3.5** and the following Python libraries installed:

- [Jupyter](http://jupyter.org/)
- [NumPy](http://www.numpy.org/)
- [SciPy](https://www.scipy.org/)
- [scikit-learn](http://scikit-learn.org/)
- [TensorFlow](http://tensorflow.org)
- [Matplotlib](http://matplotlib.org/)
- [Pandas](http://pandas.pydata.org/) (Optional)

Run this command at the terminal prompt to install [OpenCV](http://opencv.org/). Useful for image processing:

- `conda install -c https://conda.anaconda.org/menpo opencv3`

### Dataset

1. [Download the dataset](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/581faac4_traffic-signs-data/traffic-signs-data.zip). This is a pickled dataset in which we've already resized the images to 32x32.
2. Clone the project and start the notebook.
```
git clone https://github.com/udacity/CarND-Traffic-Signs
cd CarND-Traffic-Signs
jupyter notebook Traffic_Signs_Recognition.ipynb
```

### Data Set Exploration

The data set is imported by loading the pickle (*.p) files. The data set contains training, validation and test data.
Number of training examples = 34799
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43


### Step 2: Design and Test a Model Architecture
### Pre-process the Data Set (normalization, transform_image.)
I normalized the image data with Min-Max scaling to a range of [0.1, 0.9] because this leads to a more stable convergence of weight and biases.

Used computer vision transformation techniques cv2.getRotationMatrix2D and cv2.warpAffine for Data Augmentation 


### Data Augmentation
The original data set channeled through a neural network resulted in poor accuracy ~89%. Changing the layers, hyper parameters, pre-processing techniques, resulted in minor improvements in accuracy. It seemed the network won't train well on the data set if it doesn't contain enough samples. so i generated new training data

Augmented training examples = 103167
New validation examples = 25792
Number of testing examples = 12630

### Model Architecture

My architecture is the modified LeNet :

1. Convolution layer 1. Iuput shape (32,32,3), filter shape(5,5,3,32),stride 1 and 'VALID' padding, and output shape 28x28x6.

2. Activation layer 1 with rectified activation function.

3. Maximum pooling layer 1 with 2x2 kernel, stride 2 and 'VALID' padding, output shape 14x14x32.

4. Convolution layer 2. Iuput shape (14,14,32), filter shape(5,5,3,64),stride 1 and 'VALID' padding, and output shape 10x10x64.

5. Activation layer 2 with rectified activation function.

6. Dropout 0.5 

7. Maximum pooling layer 2 with 2x2 kernel, stride 2 and 'VALID' padding, output shape 5x5x64.

8. Flatten layer. Flatten the output shape of the final pooling layer such that it's 1D instead of 3D.,output 1600

9. Fully connected layer 1. This should have 120 outputs.

10. Activation layer 

11. Fully connected layer 2. This should have 84 outputs.

12. Activation layer with (dropout)

13. Fully connected layer 3. This should have 43 outputs.

The difference between my architecture and LeNet is that: 1. I add a dropout layer after the second activation layer, in order to avoid overfitting.

Hyperparameters : mu = 0, sigma = 0.1 for weight initialization, Learning rate = 0.001

Optimizer: Adam Optimzer


###  Accuracy
Validation Accuracy = 0.998
Test Accuracy = 0.949

### Output prediction
```
Image 1 - Predicted class =  17, true class = 17
Image 2 - Predicted class =  13, true class = 13
Image 3 - Predicted class =  38, true class = 38
Image 4 - Predicted class =  11, true class = 11
Image 5 - Predicted class =  33, true class = 33
Image 6 - Predicted class =  18, true class = 18
Image 7 - Predicted class =  25, true class = 25
Image 8 - Predicted class =  12, true class = 12
```

### Top 5 Softmax Probabilities For Each Image Found on the Web

top 10 softmax probabilities for the predictions 
Result : No entry  ===> 100.00%
Result : Yield  ===> 100.00%
Result : Keep right  ===> 100.00%
Result : Right-of-way at the next intersection  ===> 100.00%
Result : Roundabout mandatory  ===> 98.58%
Result : General caution  ===> 100.00%
Result : Road work  ===> 100.00% 
Result : Priority road  ===> 100.00%


