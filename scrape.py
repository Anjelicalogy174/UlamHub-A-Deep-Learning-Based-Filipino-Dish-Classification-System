from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import requests
import os

dishes = [
    "Menudo", "Adobo", "Caldereta", "Afritada", "Mechado", "Sinigang", 
    "Sisig", "Lumpia", "Kare-Kare", "Lechon", "Pancit Canton", "Pancit Malabon", 
    "Halo-Halo", "Dinuguan", "Goto", "Chicharon Bulaklak"
]

driver = webdriver.Chrome() 

# Define the base download folder
base_download_folder = os.path.join(os.path.expanduser("~"), "Downloads", "Filipino_Dishes_Images")

# Create download directory 
if not os.path.exists(base_download_folder):
    os.makedirs(base_download_folder)

# Function to scrape images for each dish
def scrape_images(dish, download_folder):
    print(f"Scraping images for {dish}...")
    driver.get("https://images.google.com/")

    # Find search bar, type the dish name, and submit
    search_box = driver.find_element("name", "q")
    search_box.clear()
    search_box.send_keys(dish)
    search_box.send_keys(Keys.RETURN)
    
    # Scroll to load more images
    time.sleep(2)
    image_urls = set()
    for _ in range(3):  # Adjust scroll range if needed
        driver.execute_script("window.scrollBy(0, 1000);")
        time.sleep(2)

        # Extract image URLs
        images = driver.find_elements("css selector", "img")
        for img in images:
            src = img.get_attribute("src")
            if src and src.startswith("http"):
                image_urls.add(src)

    # Download images
    for i, url in enumerate(image_urls):
        try:
            response = requests.get(url, stream=True)
            file_path = os.path.join(download_folder, f"{dish.replace(' ', '_')}_image_{i + 1}.jpg")
            with open(file_path, "wb") as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            print(f"Downloaded {file_path}")
        except Exception as e:
            print(f"Failed to download {url}: {e}")

# Scrape images for each dish
for dish in dishes:
    dish_folder = os.path.join(base_download_folder, dish.replace(" ", "_"))
    if not os.path.exists(dish_folder):
        os.makedirs(dish_folder)
    scrape_images(dish, dish_folder)

driver.quit()

print("Image scraping complete!")
