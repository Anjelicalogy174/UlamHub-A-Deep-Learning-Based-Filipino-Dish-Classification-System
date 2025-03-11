import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

def create_account():
    username = username_entry.get()
    password = password_entry.get()
    confirm_password = confirm_password_entry.get()

    if not username or not password:
        messagebox.showerror("Error", "Username and password cannot be empty.")
        return

    if password != confirm_password:
        messagebox.showerror("Error", "Passwords do not match.")
        return

    try:
        # Save username and password to a text file
        with open("users.txt", "a") as file:
            file.write(f"{username}:{password}\n")

        messagebox.showinfo("Success", f"Account for {username} created successfully!")
        back_to_login()
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

def back_to_login():
    root.destroy()  # Close the sign-up window

# Create the main window
root = tk.Tk()
root.title("Sign Up")
root.geometry("360x640")  # Mobile size dimensions

# Apply a modern style
style = ttk.Style()
style.configure("TFrame", background="white", width=100, height=100)
style.configure("TLabel", background="white", font=("Arial", 10))
style.configure("Header.TLabel", font=("Arial", 18, "bold"))

# Sign-up Frame
signup_frame = ttk.Frame(root)
signup_frame.pack(fill=tk.BOTH, expand=True)

signup_label = ttk.Label(signup_frame, text="Sign Up", style="Header.TLabel")
signup_label.pack(pady=10)

username_label = ttk.Label(signup_frame, text="Username:")
username_label.pack(pady=5)
username_entry = ttk.Entry(signup_frame)
username_entry.pack(pady=5)

password_label = ttk.Label(signup_frame, text="Password:")
password_label.pack(pady=5)
password_entry = ttk.Entry(signup_frame, show="*")
password_entry.pack(pady=5)

confirm_password_label = ttk.Label(signup_frame, text="Confirm Password:")
confirm_password_label.pack(pady=5)
confirm_password_entry = ttk.Entry(signup_frame, show="*")
confirm_password_entry.pack(pady=5)

signup_button = ttk.Button(signup_frame, text="Sign Up", command=create_account)
signup_button.pack(pady=10)

back_button = ttk.Button(signup_frame, text="Back to Login", command=back_to_login)
back_button.pack(pady=5)

# Run the application
root.mainloop()
