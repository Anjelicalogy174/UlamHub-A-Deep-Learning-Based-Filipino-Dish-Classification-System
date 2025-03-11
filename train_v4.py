import pandas as pd
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import shutil

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

# Define the base directory where you want to save the images
base_dir = r"C:\Users\Anjel\Documents\GitHub\UlamHub"  # Your base directory for data
train_dir = os.path.join(base_dir, 'train')  # Directory to store training images
val_dir = os.path.join(base_dir, 'validation')  # Directory to store validation images

# Create directories for training and validation if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Create subdirectories for each dish type in the train and val directories
for dish in unique_dishes:
    os.makedirs(os.path.join(train_dir, dish), exist_ok=True)
    os.makedirs(os.path.join(val_dir, dish), exist_ok=True)

# Create a mapping for dish labels
image_paths_train = [image_paths[i] for i in range(len(image_paths)) if labels[i] in y_train]
image_paths_val = [image_paths[i] for i in range(len(image_paths)) if labels[i] in y_val]

def move_images(image_paths, labels, dest_dir):
    for img_path, label in zip(image_paths, labels):
        dish = unique_dishes[label]  # Get the dish name based on the label
        dest_folder = os.path.join(dest_dir, dish)  # Directory for the dish
        
        # Move image to its corresponding folder
        try:
            img_name = os.path.basename(img_path)
            dest_path = os.path.join(dest_folder, img_name)
            shutil.copy(img_path, dest_path)  # Copy the image to the folder
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")

# Use labels (integer indices) to filter images for training and validation datasets
move_images([image_paths[i] for i in range(len(image_paths)) if labels[i] in y_train], y_train, train_dir)
move_images([image_paths[i] for i in range(len(image_paths)) if labels[i] in y_val], y_val, val_dir)
print("Images have been moved successfully.")

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(img_height, img_width, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(len(unique_dishes), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the model
history = model.fit(
    X_train, y_train_onehot,
    validation_data=(X_val, y_val_onehot),
    epochs=5,
    batch_size=32
)

# Save the model
model.save("model/ulamhub_trained_model_v2.h5")
print("Model saved!")
