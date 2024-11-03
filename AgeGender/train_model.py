import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical, Sequence
import numpy as np
import os

# Define age mapping
age_mapping = {
    '(0, 2)': 0,
    '(4, 6)': 1,
    '(8, 12)': 2,
    '(15, 20)': 3,
    '(25, 32)': 4,
    '(38, 43)': 5,
    '(48, 53)': 6,
    '(60, 100)': 7
}

# Load MobileNetV2 without the top layer
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False

# Add new layers for age and gender prediction
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Gender prediction output layer (binary classification for male and female)
gender_output = Dense(1, activation='sigmoid', name='gender_output')(x)

# Age prediction output layer (8 age groups)
age_output = Dense(8, activation='softmax', name='age_output')(x)  # 8 age groups

# Define the new model using the base model's input and the two output layers
model = Model(inputs=base_model.input, outputs=[gender_output, age_output])

# Compile the model (optimizer, loss function, and metrics)
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=['binary_crossentropy', 'categorical_crossentropy'],
    metrics=['accuracy', 'accuracy']
)

# Image data generator for training with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

# Custom data generator to handle age and gender labels
def custom_data_generator(generator):
    while True:
        images = []
        age_labels = []
        gender_labels = []

        while len(images) < generator.batch_size:
            try:
                batch_images = next(generator)
            except StopIteration:
                generator.reset()
                batch_images = next(generator)
            
            batch_start = generator.batch_index * generator.batch_size
            batch_end = batch_start + generator.batch_size
            paths = generator.filepaths[batch_start:batch_end]

            for img, path in zip(batch_images, paths):
                parts = path.split(os.path.sep)[-3:]
                if len(parts) < 3:
                    continue  # 폴더 구조가 잘못된 경우 건너뛰기
                age_folder = parts[0]
                gender_folder = parts[1].lower()

                age = age_mapping.get(age_folder, -1)
                gender = 0 if gender_folder == 'female' else 1 if gender_folder == 'male' else -1

                if age == -1 or gender == -1:
                    continue  # 유효하지 않은 라벨인 경우 건너뛰기

                images.append(img)
                age_labels.append(age)
                gender_labels.append(gender)

                if len(images) == generator.batch_size:
                    break

        age_labels = np.array(age_labels, dtype='int32')
        gender_labels = np.array(gender_labels, dtype='float32').reshape(-1, 1)
        age_labels = to_categorical(age_labels, num_classes=8).astype('float32')

        # 디버그 정보 출력
        print("Debug - Age Labels Shape:", age_labels.shape, "Data type:", age_labels.dtype)
        print("Debug - Gender Labels Shape:", gender_labels.shape, "Data type:", gender_labels.dtype)

        yield np.array(images), (gender_labels, age_labels)

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

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model.summary() 

# Train the model with both age and gender labels
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=[early_stopping]
)

# Save the trained model
model.save('age_gender_model.h5')
print("Model training complete and saved as 'age_gender_model.h5'")
