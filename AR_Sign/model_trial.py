import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# --- Load the trained model ---
model = tf.keras.models.load_model("hand_points/points.keras")

# --- Initialize MediaPipe Hands ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# --- Open webcam feed ---
cap = cv2.VideoCapture(0)

# Get model’s expected input shape (should be 42 if using 21 (x,y) points)
input_shape = model.input_shape[-1]

labels = ["A","B","C","M"]
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
            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract normalized landmark coordinates
            points = []
            for lm in hand_landmarks.landmark:
                x, y = lm.x, lm.y
                points.extend([x, y])  # 42 values total

            # Convert to numpy and reshape for model
            points_np = np.array(points, dtype=np.float32).reshape(1, -1)

            # If your model expects fewer/more points, adjust here
            if points_np.shape[1] == input_shape:
                pred = model.predict(points_np, verbose=0)
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
