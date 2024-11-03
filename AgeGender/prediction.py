import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the trained model
model = tf.keras.models.load_model('final_age_gender_model.keras')

# Image preprocessing function
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Prediction function
def predict_age_gender(image_path, model):
    processed_image = preprocess_image(image_path)
    gender_pred, age_pred = model.predict(processed_image)
    
    gender_label = 'Female' if gender_pred[0][0] < 0.5 else 'Male'
    gender_confidence = 1 - gender_pred[0][0] if gender_pred[0][0] < 0.5 else gender_pred[0][0]
    
    age_classes = {
        0: '(0, 2)',
        1: '(4, 6)',
        2: '(8, 12)',
        3: '(15, 20)',
        4: '(25, 32)',
        5: '(38, 43)',
        6: '(48, 53)',
        7: '(60, 100)'
    }
    
    age_class_index = np.argmax(age_pred[0])
    age_label = age_classes.get(age_class_index, 'Unknown')
    age_confidence = age_pred[0][age_class_index]
    
    return gender_label, gender_confidence, age_label, age_confidence

# Visualization function
def display_prediction(image_path, gender_label, gender_confidence, age_label, age_confidence):
    img = load_img(image_path, target_size=(224, 224))
    
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Gender: {gender_label} ({gender_confidence*100:.2f}% confidence)\nAge Group: {age_label} ({age_confidence*100:.2f}% confidence)", fontsize=12)
    plt.show()

# Main execution
if __name__ == "__main__":
    # Path to the image to predict
    image_path = 'images/test1.jpg'  # Replace with your image file path

    # Check if the image exists before making predictions
    if not os.path.exists(image_path):
        print(f"Error: No image file found at {image_path}")
    else:
        gender_label, gender_confidence, age_label, age_confidence = predict_age_gender(image_path, model)
        
        print(f"Gender: {gender_label} ({gender_confidence*100:.2f}%)")
        print(f"Age: {age_label} ({age_confidence*100:.2f}%)")
        
        # Display the prediction
        display_prediction(image_path, gender_label, gender_confidence, age_label, age_confidence)
