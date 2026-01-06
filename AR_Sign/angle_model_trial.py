import cv2
import mediapipe as mp
import numpy
import numpy as np
import tensorflow as tf
import math

# --- Load the trained model ---
model = tf.keras.models.load_model("hand_angles/angles.keras")

# --- Initialize MediaPipe Hands ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

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
        print(pts[prev], pts[center], pts[nxt], v1,v2, angle)
        angles.append(angle) #normalizing

    return np.array(angles)

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# --- Open webcam feed ---
cap = cv2.VideoCapture(0)

# Get model’s expected input shape (should be 16)
input_shape = model.input_shape[-1]

labels = ["D","A","B","C","E","F","G","H","I","K","L","M","N","O","P","R","S","U","W","Y"]
# --- Main loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    h, w, _ = frame.shape

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            angles = compute_hand_joint_angles_2d(hand_landmarks.landmark, frame.shape).reshape(1, 16)
            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            if angles.shape[1] == input_shape:
                print("DETECTING")
                pred = model.predict(angles, verbose=0)
                class_id = np.argmax(pred)
                confidence = np.max(pred)

                # Print on frame
                cv2.putText(frame, f"Pred: {labels[class_id]} ({confidence:.2f})",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Hand Gesture Detection", frame)

    # Exit on ESC or 'q'
    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
hands.close()
