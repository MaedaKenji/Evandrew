import cv2
import os
import numpy as np
from datetime import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


# Setup directories
save_directory = "C:\\TA\\trainCNN\\dataset"
classes = ['Maju', 'Mundur', 'Kanan', 'Kiri', 'Stop']
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
    model = tf.keras.models.Sequential([
        # First convolutional layer
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                               input_shape=(30, 200, 3), padding='same'),
        tf.keras.layers.MaxPool2D(2, 2),

        # Dropout layer added after max pooling
        # tf.keras.layers.Dropout(0.25),  # Typically, a dropout rate between 0.2 and 0.5 is used

        # Second convolutional layer
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPool2D(2, 2),

        # Dropout layer added after max pooling
        # tf.keras.layers.Dropout(0.25),  # Typically, a dropout rate between 0.2 and 0.5 is used

        # Third convolutional layer
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPool2D(2, 2),

        # Dropout layer added after max pooling
        # tf.keras.layers.Dropout(0.25),  # Typically, a dropout rate between 0.2 and 0.5 is used

        # Fourth convolutional layer
        # tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'),
        # tf.keras.layers.MaxPool2D(2,2),

        # Dropout layer added after max pooling
        # Typically, a dropout rate between 0.2 and 0.5 is used
        tf.keras.layers.Dropout(0.25),

        # Flatten the output and feed it into a dense layer
        tf.keras.layers.Flatten(),

        # Dense layer with fewer neurons
        tf.keras.layers.Dense(256, activation='relu'),

        # Dropout layer added before the output layer
        # Typically, a dropout rate between 0.2 and 0.5 is used
        tf.keras.layers.Dropout(0.5),

        # Output layer
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    # plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)

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
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)

        model = build_model()
        # model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test)) # Old fit
        model.fit(X_train, y_train, epochs=100,
                  validation_split=0.3)  # New fit
        model.save('movement_classifier.h5')  # Old save
        # model.save('movement_classifier.keras') # New save

        print("Model trained and saved as 'movement_classifier.h5'")

        plt.plot(model.history['accuracy'], label='Training Accuracy')
        plt.plot(model.history['val_accuracy'],
                 label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.ylabel('value')
        plt.xlabel('No. epoch')
        plt.legend(loc="upper left")
        plt.show()

        plt.plot(model.history['loss'], label='Training Loss')
        plt.plot(model.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.ylabel('value')
        plt.xlabel('No. epoch')
        plt.legend(loc="upper left")
        plt.show()
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
