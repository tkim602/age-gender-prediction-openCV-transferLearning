import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

# Load MobileNetV2 without the top layer
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')

# Freeze the base model to retain pre-trained weights
base_model.trainable = False

# Add new layers for age and gender prediction
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Gender output
gender_output = Dense(2, activation='softmax', name='gender_output')(x)

# Age output
age_output = Dense(len(['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']),
                   activation='softmax', name='age_output')(x)

# Define the new model
model = Model(inputs=base_model.input, outputs=[gender_output, age_output])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss={'gender_output': 'sparse_categorical_crossentropy',
                    'age_output': 'sparse_categorical_crossentropy'},
              metrics={'gender_output': 'accuracy', 'age_output': 'accuracy'})

# Image data generators for training and validation (Adience 데이터셋 경로 지정 필요)
train_datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    'path_to_adience_data',  # 이 경로를 Adience 데이터셋 위치로 변경하세요.
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse',
    subset='training',
    classes=['male', 'female'],  # 성별을 분류할 클래스 추가
    seed=42)

validation_generator = train_datagen.flow_from_directory(
    'path_to_adience_data',  # 이 경로를 Adience 데이터셋 위치로 변경하세요.
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse',
    subset='validation',
    classes=['male', 'female'],
    seed=42)

# Train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,
    steps_per_epoch=100,
    validation_steps=50
)

# Save the trained model
model.save('age_gender_model.h5')
print("Model training complete and saved as 'age_gender_model.h5'")
