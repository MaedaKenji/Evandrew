# Import necessary libraries
import tensorflow as tf
import cv2
import mediapipe as mp
import numpy as np
import datetime
import time
import serial
# from keras.models import load_model
# from bleak import BleakClient
import socket  # BT
# import asyncio#BT


# ESP32 address
# host = "192.168.4.1" # Set to ESP32 Access Point IP Address
host = "192.168.137.180"  # IP address from MSI
# host = "192.168.100.59" # 343
port = 80

kelasTemp = 'Start'
print("Global kelasTemp initialized:", kelasTemp)
classes = ['Kanan', 'Kiri', 'Maju', 'Mundur', 'Stop']
prev_frame_time = 0
idx = -1
prev_idx = -1
counter = 0
start_time = time.time()
serIsError = False

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

# Load the trained model
# model.load_weights(r'model/model_5_t.h5')
model.load_weights(r"model.h5")
# model.load_weights('model6.h5')

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Function to preprocess the image


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


try:
    ser = serial.Serial(port='COM11', baudrate=115200, timeout=1)
except:
    print("Serial error while connecting to serial port")
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

            predictions = model.predict(preprocessed_image, verbose=0)
            # Predicted class
            predicted_class = classes[np.argmax(predictions)]
            # Index class
            idx = np.argmax(predictions)
            print(f"Predicted Class: {predicted_class}, Index: {idx}")

            if idx == prev_idx:
                counter += 1  # Increment the counter when the class stays the same
            else:
                counter = 0   # Reset the counter when the class changes
                prev_idx = idx  # Update the previous index

            current_time2 = time.time()
            # Check if the predicted class has changed
            if idx >= 0:
                cv2.putText(image, f'Predicted Class: {
                            predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if idx >= 0 and serIsError == True and current_time2 - start_time >= 300:
                try:
                    ser = serial.Serial(
                        port='COM11', baudrate=115200, timeout=1)
                    serIsError = False
                except serial.SerialException as e:
                    print(f"Serial error: {e}")
                    serIsError = True
                    time.sleep(1)  # Wait before retrying
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")
                    serIsError = True
                    time.sleep(1)  # Wait before retrying

            if idx >= 0 and serIsError == False:
                if classes[np.argmax(predictions)] == 'Kanan' and counter > 5:
                    arah = 'A\n'
                    # kecepatan = 250
                    # message = f"{arah},{kecepatan}"
                    kelasTemp = 'Kanan'
                    ser.write(arah.encode("utf-8"))
                    print(f"{arah} Telah Dikirim")
                elif classes[np.argmax(predictions)] == 'Kiri' and counter > 5:
                    arah = 'E\n'
                    # kecepatan = 250
                    # message = f"{arah},{kecepatan}"
                    kelasTemp = 'Kiri'
                    ser.write(arah.encode('utf-8'))
                    print(f"{arah} Telah Dikirim")
                elif classes[np.argmax(predictions)] == 'Maju' and counter > 5:
                    arah = 'B\n'
                    # kecepatan = 250
                    # message = f"{arah},{kecepatan}"
                    kelasTemp = 'Maju'
                    ser.write(arah.encode('utf-8'))
                    print(f"{arah} Telah Dikirim")
                elif classes[np.argmax(predictions)] == 'Mundur' and counter > 5:
                    arah = 'D\n'
                    # kecepatan = 250
                    # message = f"{arah},{kecepatan}"
                    kelasTemp = 'Mundur'
                    ser.write(arah.encode('utf-8'))
                    print(f"{arah} Telah Dikirim")
                elif classes[np.argmax(predictions)] == 'Stop':
                    arah = 'C\n'
                    # kecepatan = 0
                    # message = f"{arah},{kecepatan}"
                    kelasTemp = 'Stop'
                    ser.write(arah.encode('utf-8'))
                    print(f"{arah} Telah Dikirim")

                else:
                    print(f"No Eyes Detected")

            # Calculate and display FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_frame_time)
            prev_frame_time = current_time

            # Print FPS
            # print(f"FPS: {fps} ")
            fps_text = f'FPS: {fps:.2f}'
            cv2.putText(image, fps_text, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Frame', image)

    # Display the resulting frame
    cv2.imshow('Black Frame', black_image)

    # Break the loop on 'q' key press
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
if serIsError == False:
    ser.close()
