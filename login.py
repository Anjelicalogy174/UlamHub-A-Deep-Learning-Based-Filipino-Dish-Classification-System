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
