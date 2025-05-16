import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Define constants
IMG_SIZE = (224, 224)  # Resize images
BATCH_SIZE = 32

# Load dataset (80% train, 20% validation)
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=123
)

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=123
)

# Get number of classes (breeds)
num_classes = len(train_dataset.class_names)
print(f"Number of classes: {num_classes}")  # Should print 10
print("Class names:", train_dataset.class_names)  # List of breeds


base_model = keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
base_model.trainable = False  # Freeze the base model

# Build the model
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation="softmax")  # Softmax for multi-class classification
])

# Unfreeze base model for fine-tuning
base_model.trainable = True  # Allow the model to learn features
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Model summary
model.summary()


EPOCHS = 10

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS
)

plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.show()

# Model summary
model.summary()
model.save("test.keras")