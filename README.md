# Facial Expression Recognition

 This project aims to recognize facial expressions in real-time using computer vision techniques and machine learning algorithms. The project is written in Python and uses the OpenCV and TensorFlow libraries. Thanks to this solution our humanoid robots can easily interact with people. As a future goal, recognising some specific expression will trigger an action. For instance, when Melman would see you sad he would probably try to cheer you up.

## Requirements

# To run the project, you will need the following software installed:

 - Python 3.6+
 - OpenCV 
 - TensorFlow 

## Installation

1. Clone this repository to your local machine:


> git clone https://github.com/KNR-PW/Facial-Expressions-Recognition.git


2. Install the required Python packages:


> pip install -r requirements.txt


## Usage

 To start the facial expression recognition program, run the following command:


> python main.py


 This will open up your webcam and start detecting and recognizing facial expressions in real-time. The recognized expressions will be displayed on the screen along with a rectangle around the detected face.

## Training

 The facial expression recognition model was trained on the FER-2013 dataset using a convolutional neural network (CNN) architecture. The code for training the model can be found in the `facial_expression_recognition.ipynb` file.

