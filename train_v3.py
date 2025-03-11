import pandas as pd
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Load CSV file
df = pd.read_csv(r"C:\Users\Anjel\Documents\GitHub\UlamHub\train.csv")
df.columns = df.columns.str.strip()  # Strip any leading/trailing whitespaces from column names
print("First 5 rows of the dataset:")
print(df.head())

# Define image dimensions
img_height = 224
img_width = 224

# Extract columns
image_paths = df['Image Name']
ingredients = df['Ingredients']
instructions = df['Instructions']

# Create a mapping for dish labels (based on directory names or predefined labels)
dish_names = [os.path.basename(os.path.dirname(path)) for path in image_paths]
unique_dishes = sorted(set(dish_names))
dish_to_label = {dish: idx for idx, dish in enumerate(unique_dishes)}
print("Dish to Label Mapping:")
print(dish_to_label)

# Assign labels based on dish names
labels = [dish_to_label[dish] for dish in dish_names]

# Function to load and preprocess images
def process_image(image_path):
    """
    Load and preprocess an image.
    """
    try:
        img = load_img(image_path, target_size=(img_height, img_width))  # Load image
        img_array = img_to_array(img)  # Convert to numpy array
        return img_array / 255.0  # Normalize to [0, 1]
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

# Preprocess all images and filter valid ones
valid_images, valid_labels, valid_paths = [], [], []

for img_path, label in zip(image_paths, labels):
    img = process_image(img_path)
    if img is not None:
        valid_images.append(img)
        valid_labels.append(label)
        valid_paths.append(img_path)

# Convert lists to numpy arrays
images = np.array(valid_images)
labels = np.array(valid_labels)

print(f"Total valid images: {images.shape[0]}")
print(f"Images shape: {images.shape}")
print(f"Labels shape: {labels.shape}")

# Split into training and validation sets
X_train, X_val, y_train, y_val, train_paths, val_paths = train_test_split(
    images, labels, valid_paths, test_size=0.2, random_state=42
)

# Save training and validation data splits to CSV files
def save_split_to_csv(paths, labels, filename):
    """
    Save the image paths and labels to a CSV file.
    """
    split_data = pd.DataFrame({"Image_Path": paths, "Label": labels})
    split_data.to_csv(filename, index=False)
    print(f"Saved {filename} with {len(paths)} entries.")

save_split_to_csv(train_paths, y_train, "training_data.csv")
save_split_to_csv(val_paths, y_val, "validation_data.csv")

# Convert labels to one-hot encoding
y_train_onehot = to_categorical(y_train, num_classes=len(unique_dishes))
y_val_onehot = to_categorical(y_val, num_classes=len(unique_dishes))

# Build the CNN model
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

# Train the model and save metrics
history = model.fit(
    X_train, y_train_onehot,
    validation_data=(X_val, y_val_onehot),
    epochs=5,  # Reduced epochs for faster training; adjust as needed
    batch_size=32
)

# Save the model
model.save("model/ulamhub_trained_model_v5.h5")
print("Model saved!")

# Save training metrics to a CSV file
def save_training_metrics(history, filename="training_metrics.csv"):
    """
    Save training and validation metrics (loss and accuracy) to a CSV file.
    """
    metrics = {
        "epoch": list(range(1, len(history.history['loss']) + 1)),
        "train_loss": history.history['loss'],
        "val_loss": history.history['val_loss'],
        "train_accuracy": history.history['accuracy'],
        "val_accuracy": history.history['val_accuracy'],
    }
    pd.DataFrame(metrics).to_csv(filename, index=False)
    print(f"Saved training metrics to {filename}.")

save_training_metrics(history)

# Evaluate the model on the validation set
val_predictions = model.predict(X_val)
val_predicted_classes = val_predictions.argmax(axis=1)

# Calculate evaluation metrics
accuracy = accuracy_score(y_val, val_predicted_classes)
precision = precision_score(y_val, val_predicted_classes, average="weighted")
recall = recall_score(y_val, val_predicted_classes, average="weighted")
f1 = f1_score(y_val, val_predicted_classes, average="weighted")

print("\nValidation Metrics:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Save classification report
classification_report_str = classification_report(y_val, val_predicted_classes, target_names=unique_dishes)
with open("classification_report.txt", "w") as f:
    f.write(classification_report_str)
print("\nClassification report saved to 'classification_report.txt'.")
