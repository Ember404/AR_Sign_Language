import cv2
import mediapipe as mp

# Initialize MediaPipe hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Set up webcam capture
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to capture frame")
            break

        # Flip image for mirror view and convert to RGB
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process hands
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw hand landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get wrist coordinates (Landmark 0 as an example for hand position)
                h, w, _ = frame.shape
                wrist = hand_landmarks.landmark[0]
                wrist_x, wrist_y, wrist_z = int(wrist.x * w), int(wrist.y * h), int(wrist.z)

                # Display coordinates
                cv2.putText(frame, f"Wrist: ({wrist_x}, {wrist_y},{wrist_z})", (wrist_x, wrist_y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow("Hand Tracking", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
