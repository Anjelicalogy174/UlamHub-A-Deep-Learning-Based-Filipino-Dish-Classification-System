import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

def open_recipe_window(recipe_name, recipe_description, recipe_image):
    """Opens a new window to display the selected recipe details."""
    recipe_window = tk.Toplevel()
    recipe_window.title(recipe_name)
    recipe_window.geometry("360x600")
    recipe_window.resizable(False, False)

    # Apply a style
    style = ttk.Style()
    style.configure("TFrame", background="white")
    style.configure("TLabel", background="white", font=("Arial", 10))
    style.configure("Header.TLabel", font=("Arial", 18, "bold"))

    # Main content frame
    content_frame = ttk.Frame(recipe_window)
    content_frame.pack(fill="both", expand=True, padx=10, pady=10)

    # Recipe name as a header
    recipe_label = ttk.Label(content_frame, text=recipe_name, style="Header.TLabel")
    recipe_label.pack(pady=10)

    # Recipe image
    try:
        img = Image.open(recipe_image)
        img = img.resize((300, 200))  # Adjust dimensions as needed
        img_tk = ImageTk.PhotoImage(img)

        img_label = ttk.Label(content_frame, image=img_tk)
        img_label.image = img_tk  # Keep a reference to avoid garbage collection
        img_label.pack(pady=10)
    except Exception as e:
        img_label = ttk.Label(content_frame, text="[Image Missing]", anchor="center", background="gray")
        img_label.pack(pady=10)

    # Recipe description
    desc_label = ttk.Label(content_frame, text=recipe_description, wraplength=320, anchor="center")
    desc_label.pack(pady=10)

    # Close button
    close_button = ttk.Button(content_frame, text="Close", command=recipe_window.destroy)
    close_button.pack(pady=10)

# Test function to run this file independently
if __name__ == "__main__":
    # Sample recipe data
    test_recipe_name = "Adobo"
    test_recipe_description = "A savory Filipino dish made with pork or chicken, simmered in soy sauce, vinegar, garlic, and spices."
    test_recipe_image = "images/adobo.jpg"

    root = tk.Tk()
    root.title("Recipe Viewer")
    root.geometry("360x600")

    # Test button to open a recipe page
    test_button = ttk.Button(
        root,
        text=f"Open {test_recipe_name} Recipe",
        command=lambda: open_recipe_window(test_recipe_name, test_recipe_description, test_recipe_image),
    )
    test_button.pack(pady=20)

    root.mainloop()
