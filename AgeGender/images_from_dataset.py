import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data_parent = 'dataset'  

# Load and combine fold data
folds = []
for i in range(5):  # fold_0_data.txt to fold_4_data.txt
    fold_data = pd.read_csv(os.path.join(data_parent, f'fold_{i}_data.txt'), sep='\t')
    folds.append(fold_data)

total_data = pd.concat(folds, ignore_index=True)

# Generate image paths based on folder structure
total_data['full_path'] = total_data.apply(
    lambda x: os.path.join(data_parent, 'faces', str(x.user_id), x.original_image),
    axis=1
)

# Filter out rows with missing 'age' or 'gender' values and map labels
total_data = total_data.dropna(subset=['age', 'gender'])

gender_map = {'f': 0, 'm': 1}
age_map = {
    '(0, 2)': 0,
    '(4, 6)': 1,
    '(8, 12)': 2,
    '(15, 20)': 3,
    '(25, 32)': 4,
    '(38, 43)': 5,
    '(48, 53)': 6,
    '(60, 100)': 7
}

total_data['gender'] = total_data['gender'].replace(gender_map)
total_data['age'] = total_data['age'].replace(age_map)

# Split data into training and validation sets
train_data, validation_data = train_test_split(total_data, test_size=0.2, random_state=42)

# Define a preprocessing function for the images
def preprocess_func(path, label_age, label_gender):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [128, 128]) / 255.0
    return image, {'age_output': label_age, 'gender_output': label_gender}

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((
    list(train_data['full_path']),
    (list(train_data['age']), list(train_data['gender']))
)).map(preprocess_func).batch(32).prefetch(tf.data.experimental.AUTOTUNE)

validation_dataset = tf.data.Dataset.from_tensor_slices((
    list(validation_data['full_path']),
    (list(validation_data['age']), list(validation_data['gender']))
)).map(preprocess_func).batch(32).prefetch(tf.data.experimental.AUTOTUNE)

# Define the model
inputs = tf.keras.Input(shape=(128, 128, 3))
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)

age_output = tf.keras.layers.Dense(len(age_map), activation='softmax', name='age_output')(x)
gender_output = tf.keras.layers.Dense(1, activation='sigmoid', name='gender_output')(x)

model = tf.keras.Model(inputs=inputs, outputs={'age_output': age_output, 'gender_output': gender_output})

model.compile(
    optimizer='adam',
    loss={'age_output': 'sparse_categorical_crossentropy', 'gender_output': 'binary_crossentropy'},
    metrics={'age_output': 'accuracy', 'gender_output': 'accuracy'}
)

history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=validation_dataset
)

# Visualize training history for both outputs
plt.figure(figsize=(12, 5))

# Plot age output accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['age_output_accuracy'], label='Train Age Accuracy')
plt.plot(history.history['val_age_output_accuracy'], label='Validation Age Accuracy')
plt.title('Age Prediction Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot gender output accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['gender_output_accuracy'], label='Train Gender Accuracy')
plt.plot(history.history['val_gender_output_accuracy'], label='Validation Gender Accuracy')
plt.title('Gender Prediction Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
