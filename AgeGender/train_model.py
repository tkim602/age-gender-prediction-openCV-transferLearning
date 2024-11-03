import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np

# Load MobileNetV2 without the top layer
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False

# Add new layers for age and gender prediction
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Gender output (binary classification: male or female)
gender_output = Dense(2, activation='softmax', name='gender_output')(x)

# Age output (multi-class classification for different age ranges)
age_output = Dense(8, activation='softmax', name='age_output')(x)  # Assuming 8 age groups

# Define the new model
model = Model(inputs=base_model.input, outputs=[gender_output, age_output])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss={'gender_output': 'sparse_categorical_crossentropy',
                    'age_output': 'sparse_categorical_crossentropy'},
              metrics={'gender_output': 'accuracy', 'age_output': 'accuracy'})

# Image data generator for training and validation
train_datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)

# Custom Data Generator with debugging and validation
def custom_data_generator(generator):
    while True:
        # Load a batch of images
        images = next(generator)
        batch_size = images.shape[0]  # Number of images in the batch
        
        # Initialize lists for age and gender labels
        age_labels = []
        gender_labels = []

        # Extract labels based on folder names
        for path in generator.filepaths[generator.batch_index * generator.batch_size:(generator.batch_index + 1) * generator.batch_size]:
            parts = path.split('/')[-3:]  # Extract age and gender folder names
            age_folder, gender_folder = parts[0], parts[1]

            # Assign age label based on folder name
            age_mapping = {
                '(0, 2)': 0, '(4, 6)': 1, '(8, 12)': 2,
                '(15, 20)': 3, '(25, 32)': 4, '(38, 43)': 5,
                '(48, 53)': 6, '(60, 100)': 7
            }
            age_labels.append(age_mapping.get(age_folder, -1))

            # Assign gender label based on folder name
            gender_mapping = {'female': 0, 'male': 1}
            gender_labels.append(gender_mapping.get(gender_folder, -1))

        # Convert labels to numpy arrays
        age_labels = np.array(age_labels)
        gender_labels = np.array(gender_labels)

        # Debugging: Print labels and validate range
        print("Debug - Age Labels:", age_labels)
        print("Debug - Gender Labels:", gender_labels)
        
        # Validation - Check if labels are within expected range
        if any(g < 0 or g > 1 for g in gender_labels):
            raise ValueError("Gender label out of bounds detected.")
        if any(a < 0 or a > 7 for a in age_labels):
            raise ValueError("Age label out of bounds detected.")

        yield images, {'gender_output': gender_labels, 'age_output': age_labels}

# Training generator
train_generator = train_datagen.flow_from_directory(
    'organized_data',
    target_size=(224, 224),
    batch_size=32,
    class_mode=None,
    subset='training',
    seed=42
)

# Validation generator
validation_generator = train_datagen.flow_from_directory(
    'organized_data',
    target_size=(224, 224),
    batch_size=32,
    class_mode=None,
    subset='validation',
    seed=42
)

# Apply custom generator to separate labels
train_gen = custom_data_generator(train_generator)
val_gen = custom_data_generator(validation_generator)

# Train the model
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# Save the trained model
model.save('age_gender_model.h5')
print("Model training complete and saved as 'age_gender_model.h5'")
