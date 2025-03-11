import tkinter as tk
from tkinter import filedialog, messagebox
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image, ImageTk

# Load the trained model
model = load_model("model/ulamhub_trained_model.h5")
print("Model loaded successfully!")

# Define image dimensions
img_height = 224
img_width = 224

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

# Reverse mapping for dish names
label_to_dish = {v: k for k, v in dish_to_label.items()}

# Function to process the uploaded image
def process_image(image_path):
    try:
        img = load_img(image_path, target_size=(img_height, img_width))  # Load and resize image
        img_array = img_to_array(img)  # Convert image to array
        img_array = img_array / 255.0  # Normalize image to [0, 1]
        return np.expand_dims(img_array, axis=0)  # Add batch dimension
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None  # Return None if image cannot be loaded

# Function to predict the dish
def predict_dish(image_path):
    img = process_image(image_path)

    if img is not None:
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction, axis=1)
        predicted_dish = label_to_dish[predicted_class[0]]
        return predicted_dish
    else:
        return None

# Function to open file dialog and upload an image
def upload_image():
    image_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    
    if image_path:
        try:
            # Show image on the UI
            image = Image.open(image_path)
            image = image.resize((250, 250))  # Resize image to fit UI
            photo = ImageTk.PhotoImage(image)
            label_image.config(image=photo)
            label_image.image = photo  # Keep reference to the image

            # Get prediction
            predicted_dish = predict_dish(image_path)
            if predicted_dish:
                label_result.config(text=f"Predicted Dish: {predicted_dish}")
            else:
                label_result.config(text="Error predicting the dish.")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading the image: {str(e)}")

# Create the main application window
root = tk.Tk()
root.title("UlamHub")
root.geometry("400x500")

# Upload button
btn_upload = tk.Button(root, text="Upload Image", command=upload_image)
btn_upload.pack(pady=20)

# Label for displaying uploaded image
label_image = tk.Label(root)
label_image.pack(pady=20)

# Label for displaying prediction result
label_result = tk.Label(root, text="Predicted Dish: ", font=("Helvetica", 14))
label_result.pack(pady=10)

# Start the UI loop
root.mainloop()