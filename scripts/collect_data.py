import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd

# Settings
LABEL = "hello"  # ✏️ CHANGE THIS FOR EACH GESTURE
SEQ_LENGTH = 40
CSV_PATH = "data/processed/gesture_data.csv"

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Create window
cap = cv2.VideoCapture(0)
sequence = []

print(f"Recording for label: {LABEL}. Press 'r' to record sequence, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract 21 keypoints (x, y, z)
            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])

            if len(sequence) < SEQ_LENGTH:
                sequence.append(keypoints)
            elif len(sequence) == SEQ_LENGTH:
                # Save to CSV
                flat = np.array(sequence).flatten().tolist()  # Convert to list
                row = [LABEL] + flat

                if not os.path.exists(CSV_PATH):
                    n_feats = len(flat)
                    cols = ["label"] + [f"f{i}" for i in range(n_feats)]
                    pd.DataFrame(columns=cols).to_csv(CSV_PATH, index=False)

                pd.DataFrame([row]).to_csv(CSV_PATH, mode="a", header=False, index=False)
                print(f"✅  Saved sequence for '{LABEL}'")
                sequence = []

    # Display info
    cv2.putText(frame, f"Recording: {LABEL}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    cv2.imshow("Collecting Gesture Data", frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('r'):
        print("⏺️ Recording started")

cap.release()
cv2.destroyAllWindows()
