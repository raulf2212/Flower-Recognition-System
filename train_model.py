import os
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2


base_dir = 'dataset/'
train_dir = os.path.join(base_dir, 'train')
valid_dir = os.path.join(base_dir, 'valid')

if not os.path.exists(train_dir) or not os.path.exists(valid_dir):
    print("ERROR: Dataset folders not found! Please run 'load_dataset.py' first.")
    exit()

img_size = 224
batch = 64

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch,
    class_mode='sparse'
)

test_generator = test_datagen.flow_from_directory(
    valid_dir,
    target_size=(img_size, img_size),
    batch_size=batch,
    class_mode='sparse'
)

class_indices_map = {v: k for k, v in train_generator.class_indices.items()}
with open('keras_indices_to_folder.json', 'w') as f:
    json.dump(class_indices_map, f)
print("Saved keras_indices_to_folder.json for the prediction script.")

base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = Sequential([
    Input(shape=(224, 224, 3)),
    tf.keras.layers.Rescaling(scale=2.0, offset=-1.0),
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.2),
    Dense(102, activation="softmax")
])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

epochs = 15
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=test_generator
)

model.save('Model.h5')
print("Model training complete and saved as Model.h5")