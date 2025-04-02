import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your pre-trained model
model = load_model('broken_road_model.h5')  # Replace with your model file name

# Function to preprocess the image
def preprocess_image(image):
    resized_image = cv2.resize(image, (150, 150))  # Resize to match the model's expected input size
    normalized_image = resized_image / 255.0      # Normalize pixel values to [0, 1]
    return np.expand_dims(normalized_image, axis=0)  # Add batch dimension (1, 150, 150, 3)

# Start video capture
video = cv2.VideoCapture(0)

while True:
    success, img = video.read()
    if not success:
        print("Failed to read video feed")
        break

    # Preprocess the frame
    preprocessed_frame = preprocess_image(img)

    # Predict road condition
    prediction = model.predict(preprocessed_frame)
    label = "Broken Road" if prediction[0][0] > 0.5 else "Good Road"
    confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]

    # Display the label and confidence on the video feed
    cv2.putText(img, f"{label} ({confidence:.2f})", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the video feed with annotations
    cv2.imshow("Broken Road Detection", img)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
video.release()
cv2.destroyAllWindows()
