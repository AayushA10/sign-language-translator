import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import mediapipe as mp
from collections import deque

# Load model and labels
model = tf.keras.models.load_model('models/lstm_model.h5')
labels = np.load('data/processed/label_classes.npy')

# Constants
SEQ_LENGTH = 40
seq_buffer = deque(maxlen=SEQ_LENGTH)

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

st.title("🧠 Real-Time Sign Language Gesture Recognition (LSTM + Streamlit)")
run = st.checkbox("Start Webcam")

FRAME_WINDOW = st.image([])

cap = None
if run:
    cap = cv2.VideoCapture(0)

while run and cap is not None:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
            seq_buffer.append(keypoints)

        if len(seq_buffer) == SEQ_LENGTH:
            input_seq = np.expand_dims(seq_buffer, axis=0)  # Shape: (1, 40, 63)
            pred = model.predict(input_seq)[0]
            idx = np.argmax(pred)
            confidence = pred[idx]
            pred_label = labels[idx]
            text = f"{pred_label} ({confidence:.2f})"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    FRAME_WINDOW.image(frame, channels='BGR')

if cap:
    cap.release()
