import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pickle
import matplotlib.pyplot as plt

# Define image dimensions (should be same as used in training)
img_height = 224
img_width = 224

# Load the trained model
model = load_model("model/ulamhub_trained_model.h5")

# Load the dish_to_label mapping
with open("dish_to_label.pkl", "rb") as f:
    dish_to_label = pickle.load(f)

# Reverse the dish_to_label mapping to map index to dish name
label_to_dish = {v: k for k, v in dish_to_label.items()}

# Define the function to preprocess the image (same as during training)
def process_image(image_path):
    try:
        img = load_img(image_path, target_size=(img_height, img_width))  # Load image
        img_array = img_to_array(img)  # Convert image to array
        return img_array / 255.0  # Normalize image to [0, 1] range
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None  # Return None if the image cannot be loaded

# Function to predict dish from an image
def predict_dish(image_path):
    # Preprocess the image
    img = process_image(image_path)
    if img is not None:
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        # Predict the dish class
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions, axis=1)
        return predicted_class[0], predictions
    else:
        print("Invalid image")
        return None, None

# Test the model with a sample image
image_path = r"C:\Users\Anjel\Documents\GitHub\UlamHub\Filipino_Dishes_Images\Adobo\Adobo_image_1.jpg"  # Replace with the path to your test image
predicted_class, predictions = predict_dish(image_path)

if predicted_class is not None:
    # Print predicted class index and corresponding dish name
    predicted_dish_name = label_to_dish[predicted_class]
    print(f"Predicted class index: {predicted_class}")
    print(f"Predicted dish name: {predicted_dish_name}")
    print(f"Predicted probabilities: {predictions}")
    
    # Optionally, if you want to display the image
    img = load_img(image_path)
    plt.imshow(img)
    plt.title(f"Predicted Class: {predicted_dish_name}")
    plt.axis('off')
    plt.show()