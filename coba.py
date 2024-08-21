import cv2
import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Setup directories
save_directory = "C:\\TA\\trainCNN\\dataset"
classes = ['maju', 'mundur', 'kanan', 'kiri', 'stop']
os.makedirs(save_directory, exist_ok=True)
for class_name in classes:
    os.makedirs(os.path.join(save_directory, class_name), exist_ok=True)

# Initialize webcam
cap = cv2.VideoCapture(0)

def save_frame(class_name, frame):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    filename = f"{timestamp}.jpg"
    file_path = os.path.join(save_directory, class_name, filename)
    cv2.imwrite(file_path, frame)
    print(f"Image saved to {file_path}")

def load_data():
    images = []
    labels = []
    for idx, class_name in enumerate(classes):
        class_dir = os.path.join(save_directory, class_name)
        for file in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file)
            image = cv2.imread(file_path)
            image = cv2.resize(image, (64, 64))  # Resize for simplicity
            images.append(image)
            labels.append(idx)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

def build_model():
    model = Sequential([
        # First convolutional layer
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(30, 200, 3), padding='same'),
        tf.keras.layers.MaxPool2D(2,2),

        # Second convolutional layer
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        tf.keras.layers.MaxPool2D(2,2),

        # Third convolutional layer
        tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        tf.keras.layers.MaxPool2D(2,2),

        # Dropout layer after third convolutional layer
        tf.keras.layers.Dropout(0.25),

        # Flatten the output and feed it into a dense layer
        tf.keras.layers.Flatten(),

        # Dense layer with 256 neurons
        tf.keras.layers.Dense(256, activation='relu'),

        # Dropout layer before the output layer
        tf.keras.layers.Dropout(0.5),

        # Output layer with 5 neurons (one for each class) and softmax activation
        tf.keras.layers.Dense(5, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


model = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Webcam', frame)

    key = cv2.waitKey(1) & 0xFF

    # Save frame based on key pressed
    if key == ord('1'):
        save_frame('maju', frame)
    elif key == ord('2'):
        save_frame('mundur', frame)
    elif key == ord('3'):
        save_frame('kanan', frame)
    elif key == ord('4'):
        save_frame('kiri', frame)
    elif key == ord('5'):
        save_frame('stop', frame)
    elif key == ord('k'):
        print("Training the model...")
        X, y = load_data()
        y = to_categorical(y, num_classes=len(classes))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = build_model()
        model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
        model.save('movement_classifier.h5')
        print("Model trained and saved as 'movement_classifier.h5'")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
