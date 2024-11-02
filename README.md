# Age and Gender Prediction

This project is an age and gender prediction tool that uses OpenCV's Haar Cascade Classifier for face detection and a pre-trained model to estimate age and gender based on detected faces. Currently, the project leverages Caffe-based deep learning models for age and gender classification, and OpenCV for image processing and face detection.

## Features

- **Face Detection**: Uses OpenCV's Haar Cascade Classifier to detect faces in an image.
- **Age Prediction**: Predicts the age group of each detected face using a pre-trained age prediction model.
- **Gender Prediction**: Determines the gender (male or female) of each detected face.
- **Confidence Filtering**: Only displays predictions for age and gender if the model's confidence is above a certain threshold to improve reliability.

## Future Improvement

Advanced deep learning models will be used in future updates for improving age and gender predictions. These advanced models have to be fine-tuned on a larger and more diversified dataset to provide higher accuracy in predicting age across different ranges and demographic groups. This approach will involve training or transferring learning on a new deep learning model that can handle complex variations in lighting, angle, and facial expressions.

## Structure

AgeGender.py: Main script for face detection, age, and gender prediction.
age_net.caffemodel, gender_net.caffemodel: Pre-trained models for age and gender classification.
age_deploy.prototxt, gender_deploy.prototxt: Model configuration files for Caffe.
sample1.jpg: Example input image.
README.md: Project documentation.

## Dependencies

Python 3.x
OpenCV
Numpy
Caffe (for deep learning model compatibility with OpenCV's DNN module)
