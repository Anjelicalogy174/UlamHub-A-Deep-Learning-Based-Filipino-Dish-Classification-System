import tkinter as tk
from tkinter import filedialog, messagebox
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image, ImageTk
import pandas as pd

# Load the trained model
model = load_model("model/ulamhub_trained_model.h5")
print("Model loaded successfully!")

# Define image dimensions
img_height = 224
img_width = 224

# Dish to label mapping
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

# Load the CSV with ingredients and instructions
df = pd.read_csv(r"C:\Users\Anjel\Documents\GitHub\UlamHub\train.csv")
df.columns = df.columns.str.strip()  # Strip any leading/trailing whitespaces from column names

# Create dictionaries for ingredients and instructions based on dish names
dish_ingredients = {row['Image Name'].split('\\')[-2]: row['Ingredients'] for _, row in df.iterrows()}
dish_instructions = {row['Image Name'].split('\\')[-2]: row['Instructions'] for _, row in df.iterrows()}

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

# Function to predict the dish and retrieve ingredients and instructions
def predict_dish(image_path):
    img = process_image(image_path)

    if img is not None:
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction, axis=1)
        predicted_dish = label_to_dish[predicted_class[0]]
        
        # Fetch the ingredients and instructions for the predicted dish
        ingredients = dish_ingredients.get(predicted_dish, "Ingredients not found.")
        instructions = dish_instructions.get(predicted_dish, "Instructions not found.")
        
        return predicted_dish, ingredients, instructions
    else:
        return None, None, None

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
            predicted_dish, ingredients, instructions = predict_dish(image_path)
            if predicted_dish:
                label_result.config(text=f"Predicted Dish: {predicted_dish}")
                
                # Format ingredients in bullet points
                ingredient_list = ingredients.split(',')  # Split ingredients by commas
                bullet_ingredients = '\n'.join([f"• {ingredient.strip()}" for ingredient in ingredient_list])
                label_ingredients.config(text=f"Ingredients:\n{bullet_ingredients}")
                
                # Format instructions in numbered list
                instruction_list = instructions.split('.')
                numbered_instructions = '\n'.join([f"{i+1}. {instruction.strip()}" for i, instruction in enumerate(instruction_list) if instruction.strip()])
                label_instructions.config(text=f"Instructions:\n{numbered_instructions}")
            else:
                label_result.config(text="Error predicting the dish.")
                label_ingredients.config(text="")
                label_instructions.config(text="")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading the image: {str(e)}")

# Create the main application window
root = tk.Tk()
root.title("UlamHub")
root.geometry("800x600")  # Adjusted for the new layout
root.configure(bg="#f5f5f5")  # Light background for a modern look

# Load logo
logo_path = r"C:\Users\Anjel\Documents\GitHub\UlamHub\logo\Ulamhub (3).png"
logo_image = Image.open(logo_path)
logo_image = logo_image.resize((100, 50))  # Resize logo for better fit
logo_photo = ImageTk.PhotoImage(logo_image)

# Create main frames
main_frame = tk.Frame(root, bg="#f5f5f5")
main_frame.pack(fill="both", expand=True, padx=20, pady=20)

# Create left and right frames inside the main frame
left_frame = tk.Frame(main_frame, bg="#ffffff", bd=2, relief="solid", width=350, height=400)
left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

right_frame = tk.Frame(main_frame, bg="#ffffff", bd=2, relief="solid", width=400, height=400)
right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

# Add logo to the left frame
label_logo = tk.Label(left_frame, image=logo_photo, bg="#ffffff")
label_logo.image = logo_photo  # Keep reference to the image
label_logo.grid(row=0, column=0, padx=20, pady=20)

# Upload button in the left frame
btn_upload = tk.Button(left_frame, text="Upload Image", command=upload_image, font=("Arial", 14, "bold"), bg="#6200ea", fg="white", relief="flat", padx=20, pady=10)
btn_upload.grid(row=1, column=0, pady=20)

# Image display in the left frame
label_image = tk.Label(left_frame, bg="#f5f5f5")
label_image.grid(row=2, column=0, pady=20)

# Prediction result in the right frame
label_result = tk.Label(right_frame, text="Predicted Dish: ", font=("Helvetica", 16, "bold"), wraplength=400, anchor="w", bg="#ffffff")
label_result.grid(row=0, column=0, pady=10, padx=20)

# Ingredients label in the right frame
label_ingredients = tk.Label(right_frame, text="Ingredients: ", font=("Arial", 12), wraplength=400, anchor="w", justify="left", bg="#ffffff")
label_ingredients.grid(row=1, column=0, pady=10, padx=20)

# Instructions label in the right frame
label_instructions = tk.Label(right_frame, text="Instructions: ", font=("Arial", 12), wraplength=400, anchor="w", justify="left", bg="#ffffff")
label_instructions.grid(row=2, column=0, pady=10, padx=20)

# Start the UI loop
root.mainloop()
