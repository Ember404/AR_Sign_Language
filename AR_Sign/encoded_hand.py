from time import sleep

import mediapipe as mp
import cv2
import numpy as np
import math
import keyboard

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

def angle_2d(v1, v2):
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return 0.0
    cos_theta = np.clip(dot / norm, -1.0, 1.0)
    return math.acos(cos_theta)

def compute_hand_joint_angles_2d(landmarks, image_shape):
    h, w = image_shape[:2]
    pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks])

    # define joint triplets
    angle_triplets = [
        (2, 1, 3), (3, 2, 4), #thumb
        (5, 0, 6), (6, 5, 7), (7, 6, 8), #index
        (9, 0, 10), (10, 9, 11), (11, 10, 12), #middle
        (13, 0, 14), (14, 13, 15), (15, 14, 16), #ring
        (17, 0, 18), (18, 17, 19), (19, 18, 20), #pinky
        (2,0,17), #palm width
    ]

    angles = []
    for center, prev, nxt in angle_triplets:
        v1 = pts[prev] - pts[center]
        v2 = pts[nxt] - pts[center]
        angle = angle_2d(v1, v2)
        angles.append(angle) #normalizing

    return np.array(angles)

cap = cv2.VideoCapture(0)

letter = input("Which letter would you like to encode?\n")
while True:
    success, frame = cap.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            angles = compute_hand_joint_angles_2d(hand_landmarks.landmark, frame.shape)
            #print("Hand angles:", np.degrees(angles)[1])  # degrees if you prefer
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get wrist coordinates (Landmark 0 as an example for hand position)
            h, w, _ = frame.shape
            wrist = hand_landmarks.landmark[0]
            wrist_x, wrist_y, wrist_z = int(wrist.x * w), int(wrist.y * h), int(wrist.z)

            # Display coordinates
            cv2.putText(frame, f"Wrist: ({wrist_x}, {wrist_y},{wrist_z})", (wrist_x, wrist_y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if keyboard.is_pressed('enter'):
                with open("data.csv", 'a') as dataset_file:
                    dataset_file.write("\n" + letter)
                    for i in range(len(hand_landmarks.landmark)): #for each point
                        dataset_file.write("," + str(hand_landmarks.landmark[i].x))
                        dataset_file.write("," + str(hand_landmarks.landmark[i].y))
                    for i in range(len(angles)):
                        dataset_file.write("," + str(angles[i]))
                print(f"\"{letter}\" sign saved successfully.")
                sleep(0.5) #not to detect pressed key many times

        # Show the frame
    cv2.imshow("Hand Tracking", frame)
    #cv2.imshow('Hand', frame)




    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
