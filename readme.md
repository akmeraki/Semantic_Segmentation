# Semantic Segmentation

<p align="center">
<img src="https://github.com/akmeraki/Semantic_Segmentation/blob/master/images/frankfurt_000000_000294_gtFine_color.png">
</p>


### Overview
The objective of this project is to clone human driving behavior by use of Convolutional Neural Network in a simulated driving application, we are going to use a Udacity developed Car Simulator using Unity game engine. The simulator consists of training and Autonomous modes to rive the car. First, during the training session, we will navigate our car inside the simulator using the keyboard or a joystick for continuous motion of the steering angle. While we navigating the car the simulator records training images and respective steering angles. Then we use those recorded data to train our neural network. It uses the trained model to predict steering angles for a car in the simulator given a frame from the central camera.


### Dependencies

This project requires **Python 3.6** and the following Python libraries installed:
Please utilize the environment file to install related packages.

- [Environment File](https://github.com/akmeraki/Behavioral-Cloning-Udacity/tree/master/Environment)
- [Keras](https://keras.io/)
- [NumPy](http://www.numpy.org/)
- [Scikit-learn](http://scikit-learn.org/)
- [Tensorflow](https://www.tensorflow.org/)
- [OpenCV2](http://opencv.org/)
- [Scipy](https://www.scipy.org)

### Files in this repo
- `Semantic_Segmentation.ipynb` - The jupyter notebook used to build and train the model.
- `Notes folder` - Contains the Notes taken, Paper summaries and Documents taken.
- `Image folder` - Contains the samples images of the training data and model .
- `helper.py` - python program for images pre- and  post- processing.
- `cityscapes.ipynb` - Jupyter notebook with some visualization and preprocessing of the Cityscape dataset. Please, see the notebook for correct dataset directories placing.
- `helper_cityscapes.py` - python program for images pre- and  post- processing for the Cityscape dataset.

### Architecture
<p align="center">
<img src="https://github.com/akmeraki/Semantic_Segmentation/blob/master/images/fcn_arch_vgg16.png">
</p>

A Fully Convolutional Network (FCN-8 Architecture developed at Berkeley, see [paper](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) ) was applied for the project. It uses VGG16 pretrained on ImageNet as an encoder.
Decoder is used to upsample features, extracted by the VGG16 model, to the original image size. The decoder is based on transposed convolution layers.

The goal is to assign each pixel of the input image to the appropriate class (road, backgroung, etc). So, it is a classification problem, that is why, cross entropy loss was applied.


### Model parameters
<p align="center">
<img src="">
</p>

### How to Run the Model
You can clone the repository. Then install the required dependencies. Open a jupyter lab or jupyter notebook console and Implement the notebook- `Semantic_Segmentation.ipynb`.


### About the Dataset
This Model was trained using a dataset size of 5000 images.
Link to the cityscapes Dataset: https://www.cityscapes-dataset.com/examples/#fine-annotations
