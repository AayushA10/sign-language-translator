"""
Realtime ASL alphabet classifier (CNN) + optional speech feedback
----------------------------------------------------------------
• Draws a 300×300 ROI box in the centre of the webcam feed
• Classifies the ROI every frame with your trained CNN
• Speaks the letter aloud when the prediction changes
"""

import os
import time
import cv2
import numpy as np
import pyttsx3
import tensorflow as tf

# ───────────────────────────────────────────────────────────────
# Configuration
MODEL_PATH  = "models/cnn_model.h5"
DATA_DIR    = "data/gesture_videos/asl_alphabet_train"
ROI_TOPLEFT = (100, 100)          # (x, y)
ROI_SIZE    = 300                 # square ROI
IMAGE_SIZE  = (64, 64)            # model input size
# ───────────────────────────────────────────────────────────────

# 1️⃣  Load model
model = tf.keras.models.load_model(MODEL_PATH)

# 2️⃣  Build class-label list (handles accidental double nesting)
def get_class_labels(root):
    inner = os.listdir(root)
    # If there’s exactly one sub-folder and nothing else, peek inside it
    if len(inner) == 1 and os.path.isdir(os.path.join(root, inner[0])):
        root = os.path.join(root, inner[0])
    return sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])

class_labels = get_class_labels(DATA_DIR)

# 3️⃣  Init text-to-speech
engine = pyttsx3.init()
prev_label = None

# 4️⃣  Start webcam
cap = cv2.VideoCapture(0)
fps_time = time.time()

print("📷  Press ‘q’ to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip for mirror effect
    frame = cv2.flip(frame, 1)

    # --- Extract ROI ---
    x, y = ROI_TOPLEFT
    roi = frame[y : y + ROI_SIZE, x : x + ROI_SIZE]
    img = cv2.resize(roi, IMAGE_SIZE)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # --- Predict ---
    pred       = model.predict(img, verbose=0)[0]
    idx        = np.argmax(pred)
    confidence = pred[idx]
    label      = class_labels[idx]

    # Speak only when label changes & confidence high enough
    if label != prev_label and confidence > 0.80:
        engine.say(label)
        engine.runAndWait()
        prev_label = label

    # --- Draw UI ---
    cv2.rectangle(frame, ROI_TOPLEFT, (x + ROI_SIZE, y + ROI_SIZE), (255, 0, 0), 2)
    cv2.putText(frame, f"{label} ({confidence:.2f})",
                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # FPS counter
    curr_time   = time.time()
    fps         = 1 / (curr_time - fps_time)
    fps_time    = curr_time
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("ASL Gesture Recognition (CNN)", frame)

    # --- Exit key ---
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
