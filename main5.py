import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import subprocess

# Function to switch between frames
def show_frame(frame):
    frame.tkraise()

# Login functionality
def login():
    username = username_entry.get()
    password = password_entry.get()

    if not username or not password:
        messagebox.showerror("Login Failed", "Please enter both username and password.")
        return

    # Check username and password from a text file
    try:
        with open("users.txt", "r") as file:
            users = file.readlines()

        valid_user = False
        for user in users:
            stored_username, stored_password = user.strip().split(":")
            if username == stored_username and password == stored_password:
                valid_user = True
                break
        
        if valid_user:
            messagebox.showinfo("Login Successful", f"Welcome, {username}!")
            show_frame(recipe_frame)  # Show the recipe page after successful login
            login_frame.grid_forget()  # Hide the login frame permanently after login
        else:
            messagebox.showerror("Login Failed", "Invalid username or password.")
    
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")


# Logout functionality
def logout():
    messagebox.showinfo("Logout", "You have successfully logged out!")  # Pop-up message
    username_entry.delete(0, tk.END)  # Clear the username entry field
    password_entry.delete(0, tk.END)  # Clear the password entry field
    show_frame(login_frame)  # Switch back to the login frame
    recipe_frame.grid_forget()  # Hide the recipe frame
    username_entry.delete(0, tk.END)
    password_entry.delete(0, tk.END)
    show_frame(login_frame)  # Show the login frame after logging out
    recipe_frame.grid_forget()  # Hide the recipe frame

# Sign-up functionality
def sign_up():
    subprocess.Popen(["python", "signup.py"])


# Upload image functionality
def upload_image():
    show_frame(recipe_frame)  # Close other tabs and return to recipe_frame
    subprocess.Popen(["python", "upload.py"])

# Open recipe functionality
def open_recipe(recipe_name):
    try:
        # Dynamically execute the Python script for the selected dish
        subprocess.Popen(["python", f"{recipe_name.lower()}.py"])
    except FileNotFoundError:
        messagebox.showerror("Error", f"Recipe file for {recipe_name} not found.")


# Create the main window
root = tk.Tk()
root.title("UlamHub")
root.geometry("400x800")  # Set a fixed window size
root.resizable(False, False)  # Disable resizing the window

# Configure the grid to resize dynamically
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

# Apply a modern style
style = ttk.Style()
style.configure("TFrame", background="white")
style.configure("TLabel", background="white", font=("Arial", 10))
style.configure("Header.TLabel", font=("Arial", 18, "bold"))
style.configure("TButton", font=("Arial", 10))

# Create frames for different views
login_frame = ttk.Frame(root)
recipe_frame = ttk.Frame(root)

for frame in (login_frame, recipe_frame):
    frame.grid(row=0, column=0, sticky="nsew")

# Create a canvas and a scrollbar for scrolling functionality
canvas = tk.Canvas(recipe_frame, height=800)  # Set canvas height to 800
scrollbar = ttk.Scrollbar(recipe_frame, orient="vertical", command=canvas.yview)
canvas.configure(yscrollcommand=scrollbar.set)

# Create a scrollable frame inside the canvas
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


### Login Frame ###
try:
    # Display header image
    header_img = Image.open("images/Ulamhub (3).png")
    header_img = header_img.resize((250, 90))
    header_img_tk = ImageTk.PhotoImage(header_img)

    header_label = ttk.Label(login_frame, image=header_img_tk)
    header_label.image = header_img_tk
    header_label.pack(pady=10)
except Exception as e:
    header_label = ttk.Label(login_frame, text="[Header Image Missing]", anchor="center", background="gray")
    header_label.pack(pady=10)

# Login form
login_label = ttk.Label(login_frame, text="Login", style="Header.TLabel")
login_label.pack(pady=5)

username_label = ttk.Label(login_frame, text="Username:")
username_label.pack(pady=5)
username_entry = ttk.Entry(login_frame)
username_entry.pack(pady=5)

password_label = ttk.Label(login_frame, text="Password:")
password_label.pack(pady=5)
password_entry = ttk.Entry(login_frame, show="*")
password_entry.pack(pady=5)

login_button = ttk.Button(login_frame, text="Login", command=login)
login_button.pack(pady=5)

signup_button = ttk.Button(login_frame, text="Sign Up", command=sign_up)
signup_button.pack(pady=5)

# Display footer image
try:
    footer_img = Image.open("images/signup.png")
    footer_img = footer_img.resize((350, 250))
    footer_img_tk = ImageTk.PhotoImage(footer_img)

    footer_label = ttk.Label(login_frame, image=footer_img_tk)
    footer_label.image = footer_img_tk
    footer_label.pack(pady=10)
except Exception as e:
    footer_label = ttk.Label(login_frame, text="[Footer Image Missing]", anchor="center", background="gray")
    footer_label.pack(pady=10)

### Recipe Frame ###

# Add "Welcome to UlamHub" label at the top
welcome_label = ttk.Label(scrollable_frame, text="Welcome to UlamHub!", style="Header.TLabel")
welcome_label.pack(pady=10)

# Add a small description below the welcome label
description_label = ttk.Label(scrollable_frame, text="A one-stop platform for anyone who loves Filipino cuisine.", font=("Arial", 10))
description_label.pack(pady=5)

try:
    # Logout button at the bottom
    logout_button = ttk.Button(scrollable_frame, text="Logout", command=logout)
    logout_button.pack(pady=10, side=tk.BOTTOM)

    browse_recipes_img = Image.open("images/image (10).png")  # Replace with your image path
    browse_recipes_img = browse_recipes_img.resize((350, 160))  # Adjust dimensions as needed
    browse_recipes_img_tk = ImageTk.PhotoImage(browse_recipes_img)

    browse_recipes_img_label = ttk.Label(scrollable_frame, image=browse_recipes_img_tk)
    browse_recipes_img_label.image = browse_recipes_img_tk  # Keep a reference to the image
    browse_recipes_img_label.pack(pady=10)

    up_label = ttk.Label(scrollable_frame, text="Upload Images", style="Header.TLabel")
    up_label.pack(pady=10)

except Exception as e:
    browse_recipes_img_label = ttk.Label(scrollable_frame, text="[Image Missing Above]", anchor="center", background="gray")
    browse_recipes_img_label.pack(pady=10)

recipe_label = ttk.Label(scrollable_frame, text="Browse Recipes", style="Header.TLabel")
upload_button = ttk.Button(scrollable_frame, text="Upload Image and Find Dish", command=upload_image, width=30)
upload_button.pack(pady=30)

recipe_label = ttk.Label(scrollable_frame, text="Browse Recipes", style="Header.TLabel")
recipe_label.pack(pady=10)

recipe_box_frame = ttk.Frame(scrollable_frame)
recipe_box_frame.pack(pady=5, padx=5, expand=True, fill="both")  # Expand the recipe container dynamically

# List of recipes with their images
recipes = [
    ("Karekare", "Description: Kare Kare is a type of Filipino stew with a rich and thick peanut sauce.", "images/karekare.jpg"),
    ("Adobo", "Description: Pork Adobo is pork cooked in soy sauce, vinegar, and garlic.", "images/adobo.jpg"),
    ("Sisig", "Description: Sisig is a Filipino dish made from pork jowl and ears (maskara), pork belly, and chicken liver", "images/sisig.jpg"),
    ("Sinigang", "Description: Sinigang na Baboy is a sour soup with pork ribs, vegetables, and tamarind-flavored broth.", "images/sinigang.jpg"),
    ("Caldereta", "Description: Caldereta is a comforting beef stew loaded with chorizo, onion, and potatoes.", "images/caldereta.jpg"),
    ("Igado", "Description: Igado is a popular Ilocano dish made from pork tenderloin and pig's innards such as liver, kidney, heart.", "images/igado.jpg"),
    ("Giniling", "Description: Giniling is ground meat with diced vegetables simmered in a short amount of time together.", "images/giniling.jpg"),
    ("Lechon", "Description: Lechón comprises a whole pig spit-roasted over charcoal and flavored with oil and spices.", "images/lechon.jpg")
]

# Configure recipe card dimensions
IMAGE_WIDTH = 150
IMAGE_HEIGHT = 100

for i, (name, desc, img_path) in enumerate(recipes):
    box_frame = ttk.Frame(recipe_box_frame, relief="solid", padding=5)
    box_frame.grid(row=i // 2, column=i % 2, padx=5, pady=5, sticky="nsew")

    # Display recipe image
    try:
        img = Image.open(img_path)
        img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        img_tk = ImageTk.PhotoImage(img)

        img_label = ttk.Label(box_frame, image=img_tk)
        img_label.image = img_tk
        img_label.pack()
    except Exception as e:
        img_label = ttk.Label(box_frame, text="[Image Missing]", anchor="center", background="gray")
        img_label.pack()

    # Recipe name as a button
    recipe_button = ttk.Button(box_frame, text=name, command=lambda n=name: open_recipe(n))
    recipe_button.pack()

    # Recipe description
    desc_label = ttk.Label(box_frame, text=desc, wraplength=140, anchor="center")
    desc_label.pack()

# Show the login frame initially
show_frame(login_frame)

# Run the application
root.mainloop()
