from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the trained model
model = load_model("model/ulamhub_trained_model.h5")
print("Model loaded successfully!")

# Define image dimensions (make sure they match the training size)
img_height = 224
img_width = 224

# Define the image path
image_path = r"C:\Users\Anjel\Documents\GitHub\UlamHub\Filipino_Dishes_Images\Afritada\Afritada_image_22.jpg"

# Dish to label mapping (from training code)
dish_to_label = {
    "Adobo": 0,
    "Afritada": 1,
    "Caldereta": 2,
    "Chicharon Bulaklak": 3,
    "Dinuguan": 4,
    "Halo-Halo": 5,
    "Kare-Kare": 6,
    "Lechon": 7,
    "Mechado": 8,
    "Menudo": 9,
    "Pancit Canton": 10,
    "Pancit Malabon": 11,
    "Sinigang": 12,
    "Sisig": 13
}

# Reverse the dish_to_label mapping to get the dish name
label_to_dish = {v: k for k, v in dish_to_label.items()}

# Process the image (similar to training)
def process_image(image_path):
    try:
        img = load_img(image_path, target_size=(img_height, img_width))  # Load and resize image
        img_array = img_to_array(img)  # Convert image to array
        img_array = img_array / 255.0  # Normalize image to [0, 1]
        return np.expand_dims(img_array, axis=0)  # Add batch dimension (for model input)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None  # Return None if image cannot be loaded

# Load and preprocess the image
img = process_image(image_path)

if img is not None:
    # Make a prediction
    prediction = model.predict(img)

    # Get the predicted class (index with highest probability)
    predicted_class = np.argmax(prediction, axis=1)

    # Get the predicted dish name
    predicted_dish = label_to_dish[predicted_class[0]]

    print(f"The model predicts this image is: {predicted_dish}")