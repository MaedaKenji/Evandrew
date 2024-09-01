import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained model (ensure your model is saved in the correct path)
model = tf.keras.models.load_model('movement_classifier.h5')

# Define the classes your model predicts
classes = ['Kanan', 'Kiri', 'Maju', 'Mundur', 'Stop']

# Function to preprocess the image for the model
def preprocess_image(image, target_size=(200, 30)):
    image = cv2.resize(image, target_size)  # Resize image to match the model's expected input size
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip the frame horizontally (optional, depends on your use case)
    frame = cv2.flip(frame, 1)

    # Preprocess the captured frame
    preprocessed_frame = preprocess_image(frame)

    # Make predictions using the model
    predictions = model.predict(preprocessed_frame, verbose=0)

    # Get the predicted class
    predicted_class = classes[np.argmax(predictions)]

    # Display the predicted class on the frame
    cv2.putText(frame, f'Predicted Class: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Webcam', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
