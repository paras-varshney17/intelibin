import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2

# ============================
# 1️⃣ Load the full dataset
# ============================
garbage_dataset = tf.keras.utils.image_dataset_from_directory(
    'garbage-dataset',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(224, 224)
)

# ============================
# 2️⃣ Normalize images [-1, 1]
# ============================
def format_image(image, label):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image, label

garbage_dataset = garbage_dataset.map(format_image)

# ============================
# 3️⃣ Shuffle the dataset
# ============================
AUTOTUNE = tf.data.AUTOTUNE
garbage_dataset = garbage_dataset.shuffle(buffer_size=1000, seed=42)

# ============================
# 4️⃣ Compute dataset size & split
# ============================
dataset_size = sum(1 for _ in garbage_dataset)
train_size = int(0.8 * dataset_size)
test_size  = dataset_size - train_size

train_dataset = garbage_dataset.take(train_size).prefetch(buffer_size=AUTOTUNE)
test_dataset  = garbage_dataset.skip(train_size).prefetch(buffer_size=AUTOTUNE)

print(f"Total batches: {dataset_size}")
print(f"Training batches: {train_size}")
print(f"Testing batches: {test_size}")


# Build the model
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

global_average_layer = keras.layers.GlobalAveragePooling2D()
prediction_layer = keras.layers.Dense(2, activation='softmax')  # Assuming 8 classes

model = keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
])

# Compile the model
model.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(),  # Use sparse version for integer labels
    metrics=['accuracy']
)

# Train the model
model.fit(train_dataset, epochs=10, validation_data=test_dataset)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_dataset)
print("Test accuracy:", test_acc)
print("Test loss:", test_loss)

# Save the model
model.save('model.h5')
