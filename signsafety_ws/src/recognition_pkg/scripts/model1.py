import pandas as pd
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

# Ignore warnings and set plot style
import warnings
warnings.filterwarnings("ignore")
plt.style.use("ggplot")

# Data path
dataset_path = "/home/milun/SignSafety/signsafety_ws/src/recognition_pkg/data/gtsrb"
train_csv_path = os.path.join(dataset_path, "Train.csv")
train_images_dir = '/home/milun/SignSafety/signsafety_ws/src/recognition_pkg/data/gtsrb/'

# Read dataset information
data = pd.read_csv(train_csv_path)
print(f"Loaded dataset with {len(data)} entries.")

# Preprocess images and labels
def preprocess_data(data, image_dir, input_shape):
    images = []
    labels = []

    for _, row in data.iterrows():
        # Construct image path
        image_path = os.path.join(image_dir, row["Path"])

        # Load and preprocess image
        img = Image.open(image_path)
        img = img.resize(input_shape)  # Resize to the input shape of the model
        img_array = np.array(img) / 255.0  # Normalize to [0, 1]
        images.append(img_array)

        # Append label
        labels.append(row["ClassId"])

    return np.array(images), np.array(labels)

# Define input shape (matching the model's expected input)
input_shape = (32, 32)  # Adjust based on dataset resolution
X, y = preprocess_data(data, train_images_dir, input_shape)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(input_shape[0], input_shape[1], 3)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),

    Dense(len(data["ClassId"].unique()), activation='softmax')  # Output layer for classification
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=32)

# Plot training history
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Save the model
model_save_path = os.path.join("models", "model1")
os.makedirs(model_save_path, exist_ok=True)
model.save(model_save_path)
print(f"Model saved to {model_save_path}")

