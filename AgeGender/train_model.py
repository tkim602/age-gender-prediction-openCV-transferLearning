import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical, Sequence
import numpy as np
import os
import matplotlib.pyplot as plt

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

gender_mapping = {
    'female': 0,
    'male': 1
}

class CustomSequence(Sequence):
    def __init__(self, generator, batch_size=32, num_classes=8, shuffle=True):
        self.generator = generator
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.filepaths = generator.filepaths
        self.on_epoch_end()

    def __len__(self):
        return len(self.filepaths) // self.batch_size

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_paths = [self.filepaths[k] for k in batch_indexes]
        batch_images = []
        age_labels = []
        gender_labels = []

        for path in batch_paths:
            img = load_img(path, target_size=(224, 224))
            img = img_to_array(img) / 255.0
            batch_images.append(img)

            parts = path.split(os.path.sep)[-3:]  # Adjust based on your directory structure
            age_folder = parts[0]
            gender_folder = parts[1].lower()

            age = age_mapping.get(age_folder, -1)
            gender = gender_mapping.get(gender_folder, -1)

            if age == -1 or gender == -1:
                raise ValueError(f"Invalid label in path: {path}")

            age_labels.append(age)
            gender_labels.append(gender)

        age_labels = to_categorical(age_labels, num_classes=self.num_classes).astype('float32')
        gender_labels = np.array(gender_labels, dtype='float32').reshape(-1, 1)

        return np.array(batch_images), (gender_labels, age_labels)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.filepaths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False  

x = base_model.output
x = GlobalAveragePooling2D()(x)

gender_output = Dense(1, activation='sigmoid', name='gender_output')(x)

age_output = Dense(8, activation='softmax', name='age_output')(x)

model = Model(inputs=base_model.input, outputs=[gender_output, age_output])

model.compile(
    optimizer=Adam(learning_rate=1e-4),  
    loss=['binary_crossentropy', 'categorical_crossentropy'],
    metrics=['accuracy', 'accuracy']
)

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=15,  
    width_shift_range=0.1,  
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_flow = train_datagen.flow_from_directory(
    'organized_data',  #
    target_size=(224, 224),
    batch_size=32,
    class_mode=None,
    subset='training',
    shuffle=False,  
    seed=42
)

validation_flow = train_datagen.flow_from_directory(
    'organized_data',  
    target_size=(224, 224),
    batch_size=32,
    class_mode=None,
    subset='validation',
    shuffle=False, 
    seed=42
)

train_sequence = CustomSequence(train_flow, batch_size=32, num_classes=8, shuffle=True)
validation_sequence = CustomSequence(validation_flow, batch_size=32, num_classes=8, shuffle=False)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
checkpoint = ModelCheckpoint('best_age_gender_model.keras', monitor='val_loss', save_best_only=True, mode='min')

callbacks = [early_stopping, reduce_lr, checkpoint]

model.summary()

history = model.fit(
    train_sequence,
    validation_data=validation_sequence,
    epochs=2, 
    callbacks=callbacks
)

model.save('final_age_gender_model.keras')
print("Model training complete and saved as 'final_age_gender_model.keras'")

def visualize_batch(sequence, num_images=9):
    images, labels = next(sequence)
    gender_labels, age_labels = labels

    plt.figure(figsize=(12, 12))
    for i in range(num_images):
        plt.subplot(3, 3, i+1)
        plt.imshow(images[i])
        gender = 'Female' if gender_labels[i][0] == 0 else 'Male'
        age = np.argmax(age_labels[i])
        plt.title(f"Gender: {gender}, Age Group: {age}")
        plt.axis('off')
    plt.show()

visualize_batch(train_sequence)
visualize_batch(validation_sequence)

def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['age_output_accuracy'], label='Training Age Accuracy')
    plt.plot(history.history['val_age_output_accuracy'], label='Validation Age Accuracy')
    plt.plot(history.history['gender_output_accuracy'], label='Training Gender Accuracy')
    plt.plot(history.history['val_gender_output_accuracy'], label='Validation Gender Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['age_output_loss'], label='Training Age Loss')
    plt.plot(history.history['val_age_output_loss'], label='Validation Age Loss')
    plt.plot(history.history['gender_output_loss'], label='Training Gender Loss')
    plt.plot(history.history['val_gender_output_loss'], label='Validation Gender Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

plot_history(history)
