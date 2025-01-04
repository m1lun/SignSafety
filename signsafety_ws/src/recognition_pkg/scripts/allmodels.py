import pandas as pd
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D

# Ignore warnings and set plot style
import warnings
warnings.filterwarnings("ignore")
plt.style.use("ggplot")

# Define dataset paths
dataset_path = "data/gtsrb"
train_csv_path = os.path.join(dataset_path, "Train.csv")
train_images_dir = os.path.join(dataset_path)

# Load dataset
data = pd.read_csv(train_csv_path)
print(f"Loaded dataset with {len(data)} entries.")

# Preprocessing function for images and labels
def preprocess_data(data, image_dir, input_shape):
    images = []
    labels = []
    for _, row in data.iterrows():
        # Load image from path
        image_path = os.path.join(image_dir, row["Path"])
        img = Image.open(image_path).resize(input_shape)  # Resize image
        img_array = np.array(img) / 255.0  # Normalize pixel values
        images.append(img_array)
        labels.append(row["ClassId"])
    return np.array(images), np.array(labels)

# Model input shape
input_shape = (32, 32)

# Preprocess data
X, y = preprocess_data(data, train_images_dir, input_shape)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

# Define a function to build multiple models
def build_model(variant):
    if variant == 1:
        # Model 1: Basic CNN (1 conv layer)
        return Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(input_shape[0], input_shape[1], 3)),
            MaxPooling2D((2, 2)),
            
            Flatten(),
            Dense(64, activation='relu'),
            Dense(len(np.unique(y)), activation='softmax')
        ])
    elif variant == 2:
        # Model 2: Deeper CNN (2 conv layers)
        return Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(input_shape[0], input_shape[1], 3)),
            MaxPooling2D((2, 2)),
            
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            
            Flatten(),
            Dense(128, activation='relu'),
            Dense(len(np.unique(y)), activation='softmax')
        ])
    elif variant == 3:
        # Model 3: CNN with Dropout (2 Conv Layers)
        return Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(input_shape[0], input_shape[1], 3)),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),

            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(len(np.unique(y)), activation='softmax')
        ])
    elif variant == 4:
        # Model 4: CNN with Global Average Pooling (2 Conv Layers)
        return Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(input_shape[0], input_shape[1], 3)),
            MaxPooling2D((2, 2)),

            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),

            GlobalAveragePooling2D(),
            Dense(256, activation='relu'),
            Dense(len(np.unique(y)), activation='softmax')
        ])

# Train and evaluate each model
histories = []
model_variants = 4
for variant in range(1, model_variants + 1):
    print(f"\nTraining Model {variant}...")
    model = build_model(variant)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=32, verbose=1)
    histories.append((variant, history))

    model_save_path = os.path.join("models", f"model{variant}")
    os.makedirs(model_save_path, exist_ok=True)
    model.save(model_save_path)
    print(f"Model {variant} saved to {model_save_path}")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

# Plot Training Accuracy
for variant, history in histories:
    ax1.plot(range(1, 11), history.history['accuracy'], label=f'Model {variant} Training Accuracy', marker='o')
ax1.set_title('Training Accuracy for Each Model', fontsize=16)
ax1.set_xlabel('Epochs', fontsize=14)
ax1.set_ylabel('Accuracy', fontsize=14)
ax1.legend(fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.6)

# Plot Validation Accuracy
for variant, history in histories:
    ax2.plot(range(1, 11), history.history['val_accuracy'], label=f'Model {variant} Validation Accuracy', marker='o')
ax2.set_title('Validation Accuracy for Each Model', fontsize=16)
ax2.set_xlabel('Epochs', fontsize=14)
ax2.set_ylabel('Accuracy', fontsize=14)
ax2.legend(fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
