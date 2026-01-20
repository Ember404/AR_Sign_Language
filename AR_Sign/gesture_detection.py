import time
from collections import deque
import numpy as np
import math
import tensorflow as tf
import mediapipe as mp
import cv2
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
    pts = np.vstack([pts, [landmarks[0].x * w, landmarks[9].y * h]])
    # define joint triplets
    angle_triplets = [
        (2, 1, 3), (3, 2, 4), #thumb
        (5, 0, 6), (6, 5, 7), (7, 6, 8), #index
        (9, 0, 10), (10, 9, 11), (11, 10, 12), #middle
        (13, 0, 14), (14, 13, 15), (15, 14, 16), #ring
        (17, 0, 18), (18, 17, 19), (19, 18, 20), #pinky
        (2,0,17), #palm width
        (0,21,9)#hand rotation
    ]

    angles = []
    for center, prev, nxt in angle_triplets:
        v1 = pts[prev] - pts[center]
        v2 = pts[nxt] - pts[center]
        angle = angle_2d(v1, v2)
        #print(pts[prev], pts[center], pts[nxt], v1,v2, angle)
        angles.append(angle) #normalizing

    return np.array(angles)

cap = cv2.VideoCapture(0)

x_history = deque(maxlen=10)

# Swipe detection thresholds
SWIPE_DISTANCE = 0.15   # normalized units
SWIPE_TIME = 0.5        # seconds

last_swipe_time = 0

labels = ["D","A","B","C","E","F","G","H","I","K","L","M","N","O","P","R","S","U","W","Y"]
model = tf.keras.models.load_model("hand_angles/angles.keras")
input_shape = model.input_shape[-1]

cooldown = 0
gesture = None

while True:
    success, frame = cap.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if cooldown:
        cooldown -= 1
        cv2.imshow("Swipe Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            angles = compute_hand_joint_angles_2d(hand_landmarks.landmark, frame.shape).reshape(1, 16)
            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            if angles.shape[1] == input_shape:
                pred = model.predict(angles, verbose=0)
                class_id = np.argmax(pred)
                confidence = np.max(pred)

            # Wrist landmark (id 0)
                if labels[class_id] == "L":
                    wrist_x = hand_landmarks.landmark[0].x
                    x_history.append((wrist_x, time.time()))

                    if len(x_history) >= 2:
                        x_start, t_start = x_history[0]
                        x_end, t_end = x_history[-1]

                        dx = x_end - x_start
                        dt = t_end - t_start

                        # Swipe left = movement from right to left
                        if dx < SWIPE_DISTANCE and dt < SWIPE_TIME:
                            if time.time() - last_swipe_time > 1:
                                print("Swipe Left Detected")
                                last_swipe_time = time.time()
                                x_history.clear()
                                cv2.putText(frame, f"Pred: Ł ({confidence:.2f})",
                                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                            1, (0, 255, 0), 2, cv2.LINE_AA)
                                cooldown = 100
                            else:
                                cv2.putText(frame, f"Pred: {labels[class_id]} ({confidence:.2f})",
                                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                            1, (0, 255, 0), 2, cv2.LINE_AA)

                else:
                    cv2.putText(frame, f"Pred: {labels[class_id]} ({confidence:.2f})",
                                (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Swipe Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()