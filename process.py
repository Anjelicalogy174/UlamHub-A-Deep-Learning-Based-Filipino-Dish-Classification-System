import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load CSV file
df = pd.read_csv(r"C:\Users\Anjel\Documents\GitHub\UlamHub\train.csv")
df.columns = df.columns.str.strip()  # Strip any leading/trailing spaces in column names
print(df.head())

# Image dimensions
img_height = 224
img_width = 224

# Extract columns
image_paths = df['Image Name']
ingredients = df['Ingredients']
instructions = df['Instructions']

# Create a mapping for dish labels (you can use dish names extracted from image paths or predefined labels)
dish_names = [os.path.basename(os.path.dirname(path)) for path in image_paths]
unique_dishes = sorted(set(dish_names))  # Get unique dish names
dish_to_label = {dish: idx for idx, dish in enumerate(unique_dishes)}
print("Dish to Label Mapping:", dish_to_label)

# Assign labels based on dish names
labels = [dish_to_label[dish] for dish in dish_names]

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

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Check the sizes of the split data
print("Training set size:", len(X_train))
print("Validation set size:", len(X_val))

# Data Augmentation (for training data)
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Validation data generator (without augmentation)
val_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Flow images in batches through the generators
train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
val_generator = val_datagen.flow(X_val, y_val, batch_size=32)
