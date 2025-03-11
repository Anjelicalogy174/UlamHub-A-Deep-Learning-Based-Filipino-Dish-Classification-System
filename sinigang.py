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
    go_back_button = ttk.Button(image_frame, text="\u25C0 Go Back", command=go_back)
    go_back_button.pack(pady=10)

    browse_recipes_img = Image.open("images/sinigang.jpg")  # Replace with your image path
    browse_recipes_img = browse_recipes_img.resize((350, 230))  # Adjust dimensions as needed
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

title_label = tk.Label(title_frame, text="Sinigang", font=("Arial", 20, "bold"))
title_label.pack()

stars_label = tk.Label(title_frame, text="\u2605 \u2605 \u2605 \u2605 \u2605", font=("Arial", 16), fg="green")
stars_label.pack()

# Description Section
description_frame = tk.Frame(scrollable_frame)
description_frame.pack(fill="x", pady=10)

description_title = tk.Label(description_frame, text="Description", font=("Arial", 14, "bold"))
description_title.pack(anchor="w", padx=10)

description_text = tk.Label(description_frame, text="Sinigang is a classic Filipino soup known for its sour and savory flavor, made with pork and tamarind as the main ingredients. It's a comforting dish perfect for any meal.", wraplength=360, justify="left")
description_text.pack(anchor="w", padx=10)

# Ingredients Section
ingredients_frame = tk.Frame(scrollable_frame)
ingredients_frame.pack(fill="x", pady=10)

ingredients_title = tk.Label(ingredients_frame, text="Ingredients", font=("Arial", 14, "bold"))
ingredients_title.pack(anchor="w", padx=10)

ingredients = [
    "1 kg pork (cut into pieces)",
    "1 onion (quartered)",
    "2 tomatoes (quartered)",
    "2-3 long green beans",
    "1 radish (sliced)",
    "1 eggplant (sliced)",
    "1 bunch kangkong (water spinach)",
    "1 pack sinigang mix or tamarind paste",
    "1-2 green chili peppers",
    "1 tbsp fish sauce",
    "Salt to taste",
    "1 liter water",
]

for i, ingredient in enumerate(ingredients):
    ingredient_var = tk.BooleanVar()
    tk.Checkbutton(ingredients_frame, text=ingredient, variable=ingredient_var, anchor="w").pack(anchor="w", padx=10)

# Cooking Instructions Section
instructions_frame = tk.Frame(scrollable_frame)
instructions_frame.pack(fill="x", pady=10)

instructions_title = tk.Label(instructions_frame, text="Cooking Instructions", font=("Arial", 14, "bold"))
instructions_title.pack(anchor="w", padx=10)

instructions_text = tk.Label(instructions_frame, text="Boil pork in water until tender, skimming off scum as it rises. Add onion, tomatoes, and radish, then cook until soft. Add eggplant, green beans, and sinigang mix or tamarind paste, and simmer for another 5 minutes. Add fish sauce, green chili, and kangkong. Adjust seasoning with salt. Serve hot with steamed rice.", wraplength=360, justify="left")
instructions_text.pack(anchor="w", padx=10)

root.mainloop()
