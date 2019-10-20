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
- Semantic_Segmentation.ipynb - The jupyter notebook used to build and train the model.
- Notes folder - Contains the Notes taken, Paper summaries and Documents taken.
- Image folder - Contains the samples images of the training sets.

### Architecture
<p align="center">
<img src="">
</p>

### How to Run the Model
You can clone the repository. Then install the required dependencies. Open a jupyter lab or jupyter notebook console and Implement the notebook- `Semantic_Segmentation.ipynb`.


### About the Dataset
This Model was trained using a Dataset size of 5000 images.
Link to the cityscapes Dataset: https://www.cityscapes-dataset.com/examples/#fine-annotations
