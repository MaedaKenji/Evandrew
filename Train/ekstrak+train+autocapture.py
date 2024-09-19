import datetime
from datetime import datetime
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from helpers import *
import os
import serial
import time
import numpy as np
import mediapipe as mp
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

# Function


def preprocess_image(image, target_size=(200, 30)):
    # Resize image to match the model's expected input size
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


def get_combined_bounding_box(landmarks, img_width, img_height):
    x_coords = [landmark.x for landmark in landmarks]
    y_coords = [landmark.y for landmark in landmarks]
    x_min, x_max = min(x_coords) * img_width, max(x_coords) * img_width
    y_min, y_max = min(y_coords) * img_height, max(y_coords) * img_height
    return int(x_min), int(y_min), int(x_max), int(y_max)


def save_frame(class_name, frame):
    if image_counts[class_name] < max_images_per_class:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = f"{timestamp}.jpg"
        file_path = os.path.join(save_directory, class_name, filename)
        cv2.imwrite(file_path, frame)
        image_counts[class_name] += 1
        print(f"Image saved to {file_path} (Total: {
              image_counts[class_name]}/{max_images_per_class})")
    else:
        print(f"{class_name} has reached the limit of {
              max_images_per_class} images.")
        active_saving[class_name] = False  # Stop saving for this class


def load_data():
    images = []
    labels = []
    for idx, class_name in enumerate(classes):
        class_dir = os.path.join(save_directory, class_name)
        for file in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file)
            image = cv2.imread(file_path)
            # image = cv2.resize(image, (64, 64))  # Resize for simplicity
            image = cv2.resize(image, (200, 30))  # Resize for simplicity
            # Optional: Normalize image to [0, 1] for ML models
            image = image.astype('float32') / 255.0
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


def initialize_image_counts():
    image_counts = {}
    for class_name in classes:
        class_folder = os.path.join(save_directory, class_name)
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)
        # Count the number of existing images in the folder
        image_counts[class_name] = len([name for name in os.listdir(
            class_folder) if os.path.isfile(os.path.join(class_folder, name))])
    return image_counts


kelasTemp = 'Start'
print("Global kelasTemp initialized:", kelasTemp)
classes = ['Kanan', 'Kiri', 'Maju', 'Mundur', 'Stop']
prev_frame_time = 0
idx = -1
prev_idx = -1
counter = 0
start_time = time.time()
serIsError = False
model = None

# Setup directories
save_directory = "C:\\TA\\trainCNN\\dataset"
classes = ['Maju', 'Mundur', 'Kanan', 'Kiri', 'Stop']
os.makedirs(save_directory, exist_ok=True)
for class_name in classes:
    os.makedirs(os.path.join(save_directory, class_name), exist_ok=True)

image_counts = initialize_image_counts()
max_images_per_class = 2000
active_saving = {class_name: False for class_name in classes}


# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Function to preprocess the image


try:
    ser = serial.Serial(port='COM8', baudrate=115200, timeout=1)
except:
    print("Setial error while connecting to serial port")
    serIsError = True


cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Process the frame with MediaPipe FaceMesh
    image_rgb = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h, w, _ = image.shape

    # Perform actions similar to your data extraction script
    # Prepare a black background
    black_image = np.zeros(image.shape, dtype=np.uint8)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Eyes Landmarks
            LEFT_EYE = [362, 382, 381, 380, 374, 373, 390,
                        249, 263, 466, 388, 387, 386, 385, 384, 398]
            RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154,
                         155, 133, 173, 157, 158, 159, 160, 161, 246]

            # Iris Landmarks
            LEFT_IRIS = [474, 475, 476, 477]
            RIGHT_IRIS = [469, 470, 471, 472]
            MID_LEFT_IRIS = [473]
            MID_RIGHT_IRIS = [468]

            # Combine all into 2 separate array
            all_eye_landmarks = [face_landmarks.landmark[i]
                                 for i in LEFT_EYE + RIGHT_EYE + LEFT_IRIS + RIGHT_IRIS]

            # Normalize Landmark Value
            mesh_points = np.array(
                [
                    np.multiply([i.x, i.y], [w, h]).astype(int)
                    for i in results.multi_face_landmarks[0].landmark
                ]
            )

            # Draw Eyelid Landmarks
            cv2.polylines(
                black_image, [mesh_points[LEFT_EYE]], True, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.polylines(
                black_image, [mesh_points[RIGHT_EYE]], True, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.polylines(image, [mesh_points[LEFT_EYE]],
                          True, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.polylines(image, [mesh_points[RIGHT_EYE]],
                          True, (0, 255, 0), 1, cv2.LINE_AA)

            # Draw Iris Landmarks on Mask
            cv2.polylines(
                black_image, [mesh_points[LEFT_IRIS]], True, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.polylines(
                black_image, [mesh_points[RIGHT_IRIS]], True, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.polylines(image, [mesh_points[LEFT_IRIS]],
                          True, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.polylines(image, [mesh_points[RIGHT_IRIS]],
                          True, (255, 255, 255), 1, cv2.LINE_AA)
            # Calculate the combined bounding box for both eyes
            combined_x_min, combined_y_min, combined_x_max, combined_y_max = get_combined_bounding_box(
                all_eye_landmarks, w, h)

            # After obtaining the cropped combined eye image...
            combined_eye_image = black_image[combined_y_min:combined_y_max,
                                             combined_x_min:combined_x_max]

            # After obtaining the cropped combined eye image...
            combined_eye_image2 = image[combined_y_min:combined_y_max,
                                        combined_x_min:combined_x_max]

            # Preprocess the image
            if combined_eye_image is not None and not combined_eye_image.size == 0:
                preprocessed_image = preprocess_image(combined_eye_image)
            else:
                # Handle the case where no eye is detected or the image is empty
                print("No eye detected or image is empty.")
                continue

            # Automatically save frames only for active classes
            if active_saving['Maju']:
                save_frame('Maju', combined_eye_image)
            if active_saving['Mundur']:
                save_frame('Mundur', combined_eye_image)
            if active_saving['Kanan']:
                save_frame('Kanan', combined_eye_image)
            if active_saving['Kiri']:
                save_frame('Kiri', combined_eye_image)
            if active_saving['Stop']:
                save_frame('Stop', combined_eye_image)

            # Calculate and display FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_frame_time)
            prev_frame_time = current_time

            # Print FPS
            # print(f"FPS: {fps} ")
            fps_text = f'FPS: {fps:.2f}'
            cv2.putText(image, fps_text, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('1'):  # Press '1' to start saving for 'Maju'
        active_saving['Maju'] = True
        print("Started saving for 'Maju'")
    elif key == ord('2'):  # Press '2' to start saving for 'Mundur'
        active_saving['Mundur'] = True
        print("Started saving for 'Mundur'")
    elif key == ord('3'):  # Press '3' to start saving for 'Kanan'
        active_saving['Kanan'] = True
        print("Started saving for 'Kanan'")
    elif key == ord('4'):  # Press '4' to start saving for 'Kiri'
        active_saving['Kiri'] = True
        print("Started saving for 'Kiri'")
    elif key == ord('5'):  # Press '5' to start saving for 'Stop'
        active_saving['Stop'] = True
        print("Started saving for 'Stop'")
    elif key == ord('q'):  # Press 'q' to quit
        break
    elif key == ord('k'):
        print("Training the model...")
        X, y = load_data()
        y = to_categorical(y, num_classes=len(classes))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3)

        train = ImageDataGenerator(
            validation_split=0.3,
            rescale=1/255,
            rotation_range=10,      # Slight rotations
            shear_range=0.1,        # Slight shearing
            brightness_range=(0.8, 1.2),  # Slight brightness changes
            fill_mode='nearest'     # Fill in new pixels after a transformation
        )

        validation = ImageDataGenerator(
            validation_split=0.3,
            rescale=1/255,
            rotation_range=10,      # Slight rotations
            shear_range=0.1,        # Slight shearing
            brightness_range=(0.8, 1.2),  # Slight brightness changes
            fill_mode='nearest'     # Fill in new pixels after a transformation
        )

        train_dataset = train.flow_from_directory(save_directory, target_size=(
            30, 200), batch_size=5, shuffle=False, class_mode='categorical', subset='training')

        validation_dataset = train.flow_from_directory(save_directory, target_size=(
            30, 200), batch_size=5, shuffle=False, class_mode='categorical', subset='validation')

        model = build_model()
        # model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test)) # Old fit
        # model.fit(X_train, y_train, epochs=100,
        #           validation_split=0.3)  # New fit

        rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=100)

        early_stop = EarlyStopping(
            monitor='val_loss',  # Metric to monitor
            # Number of epochs with no improvement after which training will be stopped
            patience=5,
            verbose=1,           # To log when training is being stopped
            # Restores model weights from the epoch with the best value of the monitored metric
            restore_best_weights=True
        )

        model_fit = model.fit(
            X_train, y_train,
            steps_per_epoch=10,
            epochs=40,
            validation_data=validation_dataset,
            # callbacks=[rlrop, early_stop],
            callbacks=[rlrop],
            verbose=2
        )  # Fit 2
        model.save('model6.h5')  # Old save
        # model.save('movement_classifier.keras') # New save
        print("Model trained and saved as 'model6.h5'")

        plt.plot(model_fit.history['accuracy'], label='Training Accuracy')
        plt.plot(model_fit.history['val_accuracy'],
                 label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.ylabel('value')
        plt.xlabel('No. epoch')
        plt.legend(loc="upper left")
        plt.show()

        plt.plot(model_fit.history['loss'], label='Training Loss')
        plt.plot(model_fit.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.ylabel('value')
        plt.xlabel('No. epoch')
        plt.legend(loc="upper left")
        plt.show()

    # Display the resulting frame
    cv2.imshow('Frame', image)

    # Display the resulting frame
    cv2.imshow('Black Frame', black_image)

cap.release()
cv2.destroyAllWindows()
