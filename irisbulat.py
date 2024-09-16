import cv2 as cv
import numpy as np
import mediapipe as mp
import math
import time

# Variables
frame_counter = 0

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh

# Eyes Landmarks
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390,
            249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154,
             155, 133, 173, 157, 158, 159, 160, 161, 246]

# Iris Landmarks
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Eyelid Most Landmarks
L_H_LEFT = [33]  # right eye rightmost landmarks
L_H_RIGHT = [133]  # right eye leftmost landmarks
R_H_LEFT = [362]  # left eye rightmost landmarks
R_H_RIGHT = [263]  # left eye leftmost landmarks


def eyesExtractor(img, right_eye_coords, left_eye_coords):

    i = 0
    ii = 0

    # converting color image to  scale image
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Convert the BGR image to RGB
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # getting the dimension of image
    dim = gray.shape

    # creating mask from gray scale dim
    # mask = np.zeros(dim, dtype=np.uint8)

    # drawing Eyes Shape on mask with white color
    # v.fillPoly(mask, [np.array(right_eye_coords, dtype=np.int32)], 255)
    # cv.fillPoly(mask, [np.array(left_eye_coords, dtype=np.int32)], 255)

    # showing the mask
    # cv.imshow('mask', mask)

    # draw eyes image on mask, where white shape is
    eyes = cv.bitwise_and(rgb_frame, rgb_frame, mask=mask)
    # change black color to gray other than eys
    # cv.imshow('Eyes Draw', eyes)
    # eyes[mask==0]=155

    # getting minium and maximum x and y  for right and left eyes
    # For Right Eye
    r_max_x = (max(right_eye_coords, key=lambda item: item[0]))[0]
    r_min_x = (min(right_eye_coords, key=lambda item: item[0]))[0]
    r_max_y = (max(right_eye_coords, key=lambda item: item[1]))[1]
    r_min_y = (min(right_eye_coords, key=lambda item: item[1]))[1]

    # For LEFT Eye
    l_max_x = (max(left_eye_coords, key=lambda item: item[0]))[0]
    l_min_x = (min(left_eye_coords, key=lambda item: item[0]))[0]
    l_max_y = (max(left_eye_coords, key=lambda item: item[1]))[1]
    l_min_y = (min(left_eye_coords, key=lambda item: item[1]))[1]

    # croping the eyes from mask
    cropped_right = eyes[r_min_y: r_max_y, r_min_x: r_max_x]
    cropped_left = eyes[l_min_y: l_max_y, l_min_x: l_max_x]

    # while i < 50:
    #    cv.imwrite('ctest'+str(ii)+'.jpg', eyes)
    #    ii=ii+1
    #    i=i+1

    # returning the cropped eyes
    return cropped_right, cropped_left

# Calculate distance using euclidian


def euclidean_distance(point1, point2):
    x1, y1 = point1.ravel()
    x2, y2 = point2.ravel()
    distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return distance

# Iris Position Orientation


def iris_position(iris_center, right_point, left_point):
    center_to_right_dist = euclidean_distance(iris_center, right_point)
    total_distance = euclidean_distance(right_point, left_point)
    ratio = center_to_right_dist/total_distance
    iris_position = ""
    if ratio <= 2.64:
        iris_position = "right"
    elif ratio > 2.94:
        iris_position = "left"
    else:
        iris_position = "center"
    return iris_position, ratio


# Initiate Video Capture
cap = cv.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as face_mesh:

    start_time = time.time()  # start time

    while True:
        frame_counter += 1  # frame counter

        ret, frame = cap.read()
        if not ret:
            break

        # Mirror frame
        frame = cv.flip(frame, 1)

        # Convert the BGR image to RGB
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Adjust frame height and width
        img_h, img_w = frame.shape[:2]

        # Process the frame with MediaPipe FaceMesh
        results = face_mesh.process(rgb_frame)

        # Create a mask
        mask = np.zeros((img_h, img_w), dtype=np.uint8)

        if results.multi_face_landmarks:
            # Normalize Landmark Value
            mesh_points = np.array(
                [
                    np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                    for p in results.multi_face_landmarks[0].landmark
                ]
            )
            # mesh_points = mesh_points.reshape((-1, 1, 2))  # Reshape for polylines

            # Draw Eyelid Landmarks
            cv.polylines(frame, [mesh_points[LEFT_EYE]],
                         True, (0, 255, 0), 1, cv.LINE_AA)
            cv.polylines(frame, [mesh_points[RIGHT_EYE]],
                         True, (0, 255, 0), 1, cv.LINE_AA)

            # Draw Eyelid Landmarks on Mask
            cv.polylines(mask, [mesh_points[LEFT_EYE]],
                         True, (255, 255, 255), 1, cv.LINE_AA)
            cv.polylines(mask, [mesh_points[RIGHT_EYE]],
                         True, (255, 255, 255), 1, cv.LINE_AA)

            # Transform Landmark Shape into Circle
            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(
                mesh_points[LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv.minEnclosingCircle(
                mesh_points[RIGHT_IRIS])

            # Transform center point into array
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)

            # Draw Iris Landmarks
            cv.circle(frame, center_left, int(l_radius),
                      (255, 0, 0), 1, cv.LINE_AA)
            cv.circle(frame, center_right, int(
                r_radius), (255, 0, 0), 1, cv.LINE_AA)

            # Draw Iris Landmarks on Mask
            cv.polylines(mask, [mesh_points[LEFT_IRIS]],
                         True, (255, 255, 255), 1, cv.LINE_AA)
            cv.polylines(mask, [mesh_points[RIGHT_IRIS]],
                         True, (255, 255, 255), 1, cv.LINE_AA)
            # cv.circle(mask, center_left, int(l_radius), (255, 255, 255), 1, cv.LINE_AA)
            # cv.circle(mask, center_right, int(r_radius), (255, 255, 255), 1, cv.LINE_AA)

            # Draw Dots in the corner of eyelid
            # cv.circle(frame, mesh_points[R_H_RIGHT][0], 3, (0, 255, 0), -1, cv.LINE_AA)
            # cv.circle(frame, mesh_points[R_H_LEFT][0], 3, (0, 0, 255), -1, cv.LINE_AA)
            # cv.circle(frame, mesh_points[L_H_RIGHT][0], 3, (255, 255, 255), -1, cv.LINE_AA)
            # cv.circle(frame, mesh_points[L_H_LEFT][0], 3, (0, 255, 255), -1, cv.LINE_AA)

            # Get the Iris Position oriented from Eyelid Lefmost and Rightmost Landmarks
            # iris_pos, ratio = iris_position(center_right, mesh_points[L_H_RIGHT], mesh_points[L_H_LEFT][0])
            iris_pos, ratio = iris_position(
                center_right, mesh_points[R_H_RIGHT], mesh_points[R_H_LEFT][0])
            # print(center_right)

            crop_right, crop_left = eyesExtractor(
                frame, mesh_points[RIGHT_EYE], mesh_points[LEFT_EYE])

            # Print the Iris Position
            cv.putText(
                frame,
                f"Right Iris Position : {iris_pos} {ratio:.2f}",
                (30, 30),
                cv.FONT_HERSHEY_DUPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv.LINE_AA,
            )

        # Calculating  frame per seconds FPS
        end_time = time.time()-start_time
        fps = frame_counter/end_time

        cv.putText(
            frame,
            f"FPS : {round(fps, 1)}",
            (30, 450),
            cv.FONT_HERSHEY_DUPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv.LINE_AA,
        )

        # Display the resulting frame
        cv.imshow("Iris Landmark Detection", frame)
        cv.imshow("Mask", mask)

        # Break the loop on 'q' key press
        key = cv.waitKey(1)
        if key == ord("q"):
            break

# Release the video capture and close all windows
cap.release()
cv.destroyAllWindows()
