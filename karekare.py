import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import subprocess

def go_back():
    # Close the current window and go back to the previous recipe frame
    root.destroy()  # Close the current window
    subprocess.Popen(["python", "main5.py"])  # Open the previous recipe page (or main.py)

def add_quantity():
    current = int(quantity_label["text"])
    quantity_label.config(text=str(current + 1))

def subtract_quantity():
    current = int(quantity_label["text"])
    if current > 1:
        quantity_label.config(text=str(current - 1))

def open_other_recipe(recipe_name):
    print(f"Open recipe: {recipe_name}")

root = tk.Tk()
root.title("Recipe Page")
root.geometry("400x800")  # Mobile-friendly dimensions

### Scrollable Frame Setup ###
canvas = tk.Canvas(root, height=800)  # Set the height to 800 pixels
scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
canvas.configure(yscrollcommand=scrollbar.set)

# Create the scrollable frame
scrollable_frame = ttk.Frame(canvas)

# Add the scrollable frame to the canvas
canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.grid(row=0, column=0, sticky="nsew")
scrollbar.grid(row=0, column=1, sticky="ns")

# Configure the scrollable frame to expand with resizing
scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

# Top Image Section
image_frame = tk.Frame(scrollable_frame)
image_frame.pack(fill="x", pady=10)

try:
    # Go Back Button (replacing Logout)
    go_back_button = ttk.Button(image_frame, text="◀ Go Back", command=go_back)
    go_back_button.pack(pady=10)

    browse_recipes_img = Image.open("images/kare2.jpeg")  # Replace with your image path
    browse_recipes_img = browse_recipes_img.resize((350, 200))  # Adjust dimensions as needed
    browse_recipes_img_tk = ImageTk.PhotoImage(browse_recipes_img)

    browse_recipes_img_label = ttk.Label(scrollable_frame, image=browse_recipes_img_tk)
    browse_recipes_img_label.image = browse_recipes_img_tk  # Keep a reference to the image
    browse_recipes_img_label.pack(pady=10)
except Exception as e:
    img_label = ttk.Label(image_frame, text="[Image Missing]", anchor="center", background="gray")
    img_label.pack()

# Title and Ratings Section
title_frame = tk.Frame(scrollable_frame)
title_frame.pack(fill="x", pady=10)

title_label = tk.Label(title_frame, text="Kare-Kare", font=("Arial", 20, "bold"))
title_label.pack()

stars_label = tk.Label(title_frame, text="★ ★ ★ ★ ★", font=("Arial", 16), fg="green")
stars_label.pack()

# Description Section
description_frame = tk.Frame(scrollable_frame)
description_frame.pack(fill="x", pady=10)

description_title = tk.Label(description_frame, text="Description", font=("Arial", 14, "bold"))
description_title.pack(anchor="w", padx=10)

description_text = tk.Label(description_frame, text="Kare-Kare is a delicious Filipino dish made with oxtail, tripe, and a rich peanut sauce. It is often paired with shrimp paste and enjoyed with vegetables like string beans and eggplant.", wraplength=360, justify="left")
description_text.pack(anchor="w", padx=10)

# Ingredients Section
ingredients_frame = tk.Frame(scrollable_frame)
ingredients_frame.pack(fill="x", pady=10)

ingredients_title = tk.Label(ingredients_frame, text="Ingredients", font=("Arial", 14, "bold"))
ingredients_title.pack(anchor="w", padx=10)

ingredients = [
    "1 kg oxtail (or pork hock)",
    "1/2 lb tripe",
    "1 bunch string beans",
    "1 eggplant (sliced)",
    "1/2 banana blossom (optional)",
    "1/4 cup peanut butter",
    "1/4 cup ground peanuts",
    "1/2 cup annatto oil (for color)",
    "1 onion (sliced)",
    "4 cloves garlic (minced)",
    "1 tbsp shrimp paste (bagoong)",
    "Salt to taste",
    "Water",
]

for i, ingredient in enumerate(ingredients):
    ingredient_var = tk.BooleanVar()
    tk.Checkbutton(ingredients_frame, text=ingredient, variable=ingredient_var, anchor="w").pack(anchor="w", padx=10)

# Cooking Instructions Section
instructions_frame = tk.Frame(scrollable_frame)
instructions_frame.pack(fill="x", pady=10)

instructions_title = tk.Label(instructions_frame, text="Cooking Instructions", font=("Arial", 14, "bold"))
instructions_title.pack(anchor="w", padx=10)

instructions_text = tk.Label(instructions_frame, text="Marinate meat with soy sauce, vinegar, garlic, and bay leaf for at least 30 minutes. Saut\u00e9 onion and garlic in oil, then add marinated meat. Add water, peppercorns, and simmer until meat is tender. Season with salt and pepper to taste.Boil oxtail and tripe until tender. Sauté onion and garlic in annatto oil until fragrant. Add the boiled oxtail and tripe to the sautéed mixture. Add peanut butter, ground peanuts, and water, then bring to a boil. Add string beans, eggplant, and banana blossom. Simmer for 5-10 minutes. Season with salt and serve with shrimp paste (bagoong) on the side.", wraplength=360, justify="left")
instructions_text.pack(anchor="w", padx=10)

root.mainloop()
