import pandas as pd
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Load CSV file
df = pd.read_csv(r"C:\Users\Anjel\Documents\GitHub\UlamHub\train.csv")
df.columns = df.columns.str.strip()  # Strip any leading/trailing whitespaces from column names
print(df.head())

# Define image dimensions
img_height = 224
img_width = 224

# Extract columns
image_paths = df['Image Name']
ingredients = df['Ingredients']
instructions = df['Instructions']

# Create a mapping for dish labels (you can use dish names extracted from image paths or predefined labels)
dish_names = [os.path.basename(os.path.dirname(path)) for path in image_paths]
unique_dishes = sorted(set(dish_names))
dish_to_label = {dish: idx for idx, dish in enumerate(unique_dishes)}
print("Dish to Label Mapping:", dish_to_label)

# Assign labels based on dish names
labels = [dish_to_label[dish] for dish in dish_names]

# Load and preprocess images
def process_image(image_path):
    try:
        print(f"Processing image: {image_path}")  # Debugging line to check the path
        img = load_img(image_path, target_size=(img_height, img_width))  # Load image
        img_array = img_to_array(img)  # Convert image to array
        return img_array / 255.0  # Normalize image to [0, 1] range
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None  # Return None if the image cannot be loaded

# Preprocess all images and keep only valid ones
valid_images = []
valid_labels = []

for img_path, label in zip(image_paths, labels):
    img = process_image(img_path)
    if img is not None:
        valid_images.append(img)
        valid_labels.append(label)

# Convert lists to numpy arrays
images = np.array(valid_images)
labels = np.array(valid_labels)

# Check if lengths match
print(f"Images shape: {images.shape}")
print(f"Labels shape: {labels.shape}")

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Check the sizes of the split data
print("Training set size:", len(X_train))
print("Validation set size:", len(X_val))

# Convert labels to one-hot encoding
y_train_onehot = to_categorical(y_train, num_classes=len(unique_dishes))
y_val_onehot = to_categorical(y_val, num_classes=len(unique_dishes))

# Load the ResNet50 model pre-trained on ImageNet, without the top layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Build the model by adding custom layers on top of ResNet50
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(len(unique_dishes), activation="softmax")
])

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Use EarlyStopping and ReduceLROnPlateau for better training control
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

# Train the model
history = model.fit(
    X_train, y_train_onehot,
    validation_data=(X_val, y_val_onehot),
    epochs=25,
    batch_size=32,
    callbacks=[early_stopping, lr_scheduler]
)

# Save the trained model
model.save("model/ulamhub_trained_resnet50.h5")
print("Model saved!")