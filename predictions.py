# Load trained model
from tensorflow.keras.models import load_model
model = load_model("model/ulamhub_trained_resnet50.h5")

# Load an image for prediction
test_image_path = r"C:\Users\Anjel\Documents\GitHub\UlamHub\Filipino_Dishes_Images\Adobo\Adobo_image_3.jpg"  # Replace with your image path
test_img = process_image(test_image_path)
test_img = np.expand_dims(test_img, axis=0)  # Add batch dimension

# Predict
predictions = model.predict(test_img)
predicted_label = np.argmax(predictions, axis=1)[0]

# Map label to dish name
predicted_dish = unique_dishes[predicted_label]
print(f"Predicted Dish: {predicted_dish}")

# Retrieve ingredients and instructions
matching_rows = df[dish_names == predicted_dish]
for idx, row in matching_rows.iterrows():
    print(f"Ingredients: {row['Ingredients']}")
    print(f"Instructions: {row['Instructions']}")