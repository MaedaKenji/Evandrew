# Import necessary libraries
import serial.tools.list_ports
import serial
import time
import tensorflow as tf
import cv2
import mediapipe as mp
import numpy as np


import logging
logging.getLogger('tensorflow').disabled = True

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
prev_class = 'Start'
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


# model.load_weights(r'model/model_5_t.h5')
model.load_weights(r'model.h5')
# model.load_weights('model6.h5')

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)


def preprocess_image(image, target_size=(200, 30)):
    image = cv2.resize(image, target_size)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image


def get_combined_bounding_box(landmarks, img_width, img_height):
    x_coords = [landmark.x for landmark in landmarks]
    y_coords = [landmark.y for landmark in landmarks]
    x_min, x_max = min(x_coords) * img_width, max(x_coords) * img_width
    y_min, y_max = min(y_coords) * img_height, max(y_coords) * img_height
    return int(x_min), int(y_min), int(x_max), int(y_max)


def list_serial_ports():
    ports = serial.tools.list_ports.comports()
    return [port.device for port in ports]


def select_port_or_skip(ports):
    print("Available serial ports:")
    for i, port in enumerate(ports):
        print(f"{i}: {port}")
    print(f"{len(ports)}: Skip serial connection")

    while True:
        try:
            choice = int(
                input("Select a port by number or type the number to skip: "))
            if 0 <= choice < len(ports):
                return ports[choice]
            elif choice == len(ports):
                return None
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")


def retry_serial_connection():
    while True:
        ports = list_serial_ports()
        if not ports:
            print("No serial ports available.")
            return None
        else:
            selected_port = select_port_or_skip(
                ports)
            if not selected_port:
                print("Proceeding without serial connection.")
                return None

        try:
            ser = serial.Serial(port=selected_port, baudrate=115200, timeout=1)
            print(f"Connected to {selected_port}")
            return ser
        except serial.SerialException:
            print("Serial error while connecting to serial port. Please try again.")


# ports = list_serial_ports()
# if not ports:
#     print("No serial ports available.")
#     selected_port = None
# else:
#     selected_port = select_port_or_skip(ports)

# ser = None
# if selected_port:
#     try:
#         ser = serial.Serial(port=selected_port, baudrate=115200, timeout=1)
#         print(f"Connected to {selected_port}")
#         serIsError = False
#     except serial.SerialException:
#         print("Serial error while connecting to serial port.")
#         serIsError = True
# else:
#     print("Proceeding without serial connection.")
#     serIsError = True

ser = retry_serial_connection()
if ser:
    serIsError = False
else:
    serIsError = True


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image_rgb = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h, w, _ = image.shape

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
                print(f"No Eyes Detected")
                if serIsError == False:
                    arah = 'C\n'
                    kelasTemp = 'Stop'
                    ser.write(arah.encode('utf-8'))
                    print(f"{arah} Telah Dikirim")

            predictions = model.predict(preprocessed_image, verbose=0)
            predicted_class = classes[np.argmax(predictions)]
            idx = np.argmax(predictions)
            # print(f"Predicted Class: {predicted_class}, Index: {idx}")

            if idx == prev_idx:
                counter += 1
            else:
                counter = 0
                prev_idx = idx

            current_time2 = time.time()
            if idx >= 0:
                cv2.putText(image, f'Predicted Class: {
                            predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


            if idx >= 0 and serIsError == False:
                if classes[np.argmax(predictions)] == 'Kanan' and prev_class == 'Stop':
                    arah = 'E\n'
                    prev_class = 'Kanan'
                    # kecepatan = 250
                    # message = f"{arah},{kecepatan}"
                    kelasTemp = 'Kanan'
                    ser.write(arah.encode("utf-8"))
                    arah_cleaned = arah.replace("\n", "")
                    print(f"{arah_cleaned} Telah Dikirim")
                elif classes[np.argmax(predictions)] == 'Kiri' and prev_class == 'Stop':
                    arah = 'A\n'
                    prev_class = 'Kiri'
                    # kecepatan = 250
                    # message = f"{arah},{kecepatan}"
                    kelasTemp = 'Kiri'
                    ser.write(arah.encode('utf-8'))
                    arah_cleaned = arah.replace("\n", "")
                    print(f"{arah_cleaned} Telah Dikirim")
                elif classes[np.argmax(predictions)] == 'Maju' and prev_class == 'Stop':
                    arah = 'B\n'
                    prev_class = 'Maju'
                    # kecepatan = 250
                    # message = f"{arah},{kecepatan}"
                    kelasTemp = 'Maju'
                    ser.write(arah.encode('utf-8'))
                    arah_cleaned = arah.replace("\n", "")
                    print(f"{arah_cleaned} Telah Dikirim")
                elif classes[np.argmax(predictions)] == 'Mundur' and prev_class == 'Stop':
                    arah = 'D\n'
                    prev_class = 'Mundur'
                    # kecepatan = 250
                    # message = f"{arah},{kecepatan}"
                    kelasTemp = 'Mundur'
                    ser.write(arah.encode('utf-8'))
                    arah_cleaned = arah.replace("\n", "")
                    print(f"{arah_cleaned} Telah Dikirim")
                elif classes[np.argmax(predictions)] == 'Stop' and counter >= 5:
                    arah = 'C\n'
                    prev_class = 'Stop'
                    counter = 0
                    # kecepatan = 0
                    # message = f"{arah},{kecepatan}"
                    kelasTemp = 'Stop'
                    ser.write(arah.encode('utf-8'))
                    arah_cleaned = arah.replace("\n", "")
                    print(f"{arah_cleaned} Telah Dikirim")

                # else:
                #     # print(f"No Eyes Detected")
                #     if serIsError == False:
                #         arah = 'C\n'
                #         kelasTemp = 'Stop'
                #         ser.write(arah.encode('utf-8'))
                #         print(f"{arah} Telah Dikirim")

            current_time = time.time()
            fps = 1 / (current_time - prev_frame_time)
            prev_frame_time = current_time

            # print(f"FPS: {fps} ")
            fps_text = f'FPS: {fps:.2f}'
            cv2.putText(image, fps_text, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Frame', image)
    cv2.imshow('Black Frame', black_image)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
if serIsError == False and ser is not None:
    ser.close()
